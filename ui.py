from __future__ import annotations

import os
import re

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("AUTO_APPROVE_HUMAN", "true")

from branding import get_display_title, get_example_analysis_prefix, get_subtitle
from llm_utils import get_llm_status
from main import run
from paths import REPORTS_DIR, save_user_report
from prompt_optimizer import (
    compare_test,
    default_template_choice,
    default_test_probe,
    example_for_mode,
    format_length_stats,
    parse_template_choice,
    run_optimize,
    save_optimized_prompt,
    templates_for_mode,
)
from skills import list_skills
from tools import count_expected_tasks

HINT_OK = "[就绪]"
HINT_WARN = "[注意]"
HINT_INFO = "[提示]"

MODE_LABELS = {
    "系统提示词": "system",
    "用户提示词": "user",
    "迭代改进": "iterate",
}


def _example_fmcg() -> str:
    return (
        get_example_analysis_prefix()
        + "1. 统计便利店、商超、电商、特通渠道销量与销售额对比；"
        "2. 输出气泡水及电解质水 SKU 销售 TOP 排行；"
        "3. 检索渠道陈列费与定价政策；"
        "4. 整合 SQL 销售数据、pandas 统计与政策文档生成周报；"
        "5. 查询核心 SKU 渠道库存；"
        "6. 调研竞品无糖气泡水市场动态。"
    )


EXAMPLE_FMCG = _example_fmcg()
EXAMPLE_SIMPLE = "统计渠道销量、输出 SKU 排行、检索陈列政策"
EXAMPLE_CHANNEL = "对比便利店、商超、电商、特通四个渠道的销售表现与份额"

EXAMPLE_TEMPLATE = """【区域/品类背景，如：华东气泡水 Q2 复盘】，需要完成：
1. 【数据类，如：统计各渠道销量对比】
2. 【排行类，如：输出 SKU TOP10】
3. 【政策类，如：检索陈列费用政策】
4. 【报告类，如：整合多源数据生成周报】
（建议 3～6 项，一次不超过 10 项）"""

SKILLS_MD = "\n".join(
    [
        "### Skill 列表",
        "",
        "| 名称 | 用途 | 触发词 |",
        "|------|------|--------|",
        *[
            f"| {s['name']} | {s['description']} | {', '.join(s['triggers'][:3])} |"
            for s in list_skills()
        ],
        "",
        "任务描述包含触发词时，优先调用对应 Skill，输出固定章节结构。",
    ]
)

WRITING_TIPS_MD = """
### 需求写法

每条任务写一件事，带上业务关键词便于匹配 Skill：

| 类型 | 示例 |
|------|------|
| 渠道分析 | 统计便利店/商超/电商渠道销量对比 |
| SKU 排行 | 输出销售额 TOP10 SKU |
| 政策核对 | 检索陈列费、定价政策 |
| 整合报告 | 整合销售数据与政策文档生成周报 |
"""

GUIDE_MD = """
1. 选择示例或自行填写分析任务（3～6 项为宜）
2. 输入框下方显示 `[就绪]` 后再点击生成
3. 等待 2～6 分钟（含 SQL/pandas 统计）
4. 在页面阅读或下载 Markdown 周报

数据：`data/analytics/channel_sales.csv` + `data/documents/` 政策文档
"""

FAQ_MD = """
**需要会编程吗？**  
不需要，用中文列出分析项即可。

**输入框提示含义？**  
`[就绪]` 表示任务已识别；`[注意]` 表示描述需补充。

**报告在哪？**  
保存在 `reports/` 目录，也可在右侧下载。

**失败怎么办？**  
检查 `.env` 中的 API Key，或将质量重试次数调为 0。
"""

OPTIMIZER_GUIDE_MD = """
### 提示词优化用法

旁路工具：改提示词、对比效果；**不会**自动改仓库里的 Agent system。

1. **系统提示词**：默认「Agent短约束」；控长度用「系统精简」；要长模板用「通用结构化」
2. **用户提示词**：默认「周报任务」→ 编号列表，方便拆分/Skill；可一键「优化并生成周报」
3. **迭代改进**：按改进需求微调；想变短选「迭代压缩」

试跑对比会显示两侧字数差。优化结果可保存到 `data/outputs/`。
"""

REPORTS_FOLDER_HINT = f"报告目录：`{REPORTS_DIR.resolve()}`"


def _is_warning(hint: str) -> bool:
    return hint.startswith(HINT_WARN)


def analyze_user_input(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return f"{HINT_INFO} 请输入分析任务，或点击示例按钮。"

    if "【" in text and "】" in text:
        return f"{HINT_INFO} 请把【】中的占位内容改成真实业务描述。"

    if len(text) < 8:
        return f"{HINT_WARN} 内容过短，请补充具体任务，如「统计渠道销量、输出 SKU 排行」。"

    vague_only = re.fullmatch(
        r"[\s\d]*(?:帮我|请帮|做一个|弄个|搞个|整一个|生成|写个)?"
        r"(?:方案|计划|报告|文档|东西|分析)?[\s\d]*",
        text,
    )
    if vague_only or (len(text) < 20 and count_expected_tasks(text) <= 1):
        return (
            f"{HINT_WARN} 描述过笼统，请列出具体分析项。\n"
            "示例：`统计渠道销量、输出 SKU 排行、检索陈列政策`"
        )

    n = count_expected_tasks(text)
    has_list_markers = bool(
        re.search(r"\d+[\.、]\s*", text) or "、" in text or "，" in text or ";" in text
    )

    if n > 10:
        return f"{HINT_WARN} 任务约 {n} 项，超过上限 10 项，请删减或分批执行。"

    if n >= 3:
        return f"{HINT_OK} 已识别 {n} 项任务，可以生成周报。"

    if n == 2:
        return f"{HINT_OK} 已识别 {n} 项任务，可生成；补充 1～2 项会更完整。"

    if n == 1 and has_list_markers:
        return f"{HINT_WARN} 只识别到 1 项，请检查分隔符（顿号或 1. 2. 3.）。"

    if n == 1:
        return (
            f"{HINT_WARN} 目前只有 1 项任务。\n"
            "多项请用顿号或编号分开，如：`渠道对比、SKU 排行、政策检索`"
        )

    return f"{HINT_INFO} 请按书写指南补充任务后再生成。"


def generate_report(user_input: str, max_retry: int) -> tuple[str, str, str, str | None]:
    if not user_input.strip():
        return "请先输入分析任务。", "等待输入", "", None

    hint = analyze_user_input(user_input)
    if _is_warning(hint) and "【" not in user_input:
        return f"请先完善需求\n\n{hint}", "等待修改", "", None

    if "【" in user_input and "】" in user_input:
        return "请完成填空模板后再生成。", "等待填写", "", None

    try:
        state = run(user_input.strip(), verbose=False, max_retry=int(max_retry))
        final_output = state.get("final_output") or "未生成内容，请重试。"
        thread_id = state.get("thread_id", "")
        saved_path = save_user_report(final_output, thread_id)

        metrics = state.get("run_metrics")
        m = metrics.model_dump() if hasattr(metrics, "model_dump") else {}

        status = (
            f"生成完成\n\n"
            f"- 分析项：{len(state.get('task_list') or [])}\n"
            f"- 完成率：{m.get('final_pass_rate', 0)}%\n"
            f"- LLM 调用：{len(m.get('llm_calls', []))} 次"
        )
        save_hint = f"已保存至 `{saved_path}`"
        return final_output, status, save_hint, str(saved_path)

    except Exception as e:
        err = f"生成失败：{e}\n请检查网络与 API Key，或将质量重试次数设为 0。"
        return f"运行失败\n\n{e}", err, "", None


def _mode_key(label: str) -> str:
    return MODE_LABELS.get(label, "system")


def on_mode_change(label: str):
    mode = _mode_key(label)
    choices = templates_for_mode(mode)  # type: ignore[arg-type]
    default = default_template_choice(mode)  # type: ignore[arg-type]
    show_req = mode == "iterate"
    probe = default_test_probe(mode)  # type: ignore[arg-type]
    return (
        gr.update(choices=choices, value=default),
        gr.update(visible=show_req),
        probe,
    )


def fill_agent_example(label: str):
    mode = _mode_key(label)
    prompt, req = example_for_mode(mode)  # type: ignore[arg-type]
    probe = default_test_probe(mode)  # type: ignore[arg-type]
    return prompt, req, probe, f"已填入 {label} 示例"


def do_optimize(label: str, template_choice: str, prompt: str, requirements: str):
    mode = _mode_key(label)
    tid = parse_template_choice(template_choice) or None
    try:
        out = run_optimize(prompt, mode, tid, requirements)  # type: ignore[arg-type]
        stats = format_length_stats(prompt, out)
        tip = ""
        if mode == "user":
            tip = "；可「应用到周报」或「优化并生成周报」"
        elif mode == "system":
            tip = "；系统结果需手动贴回 agents.py，不会自动改代码"
        return out, f"优化完成（{label} / {tid or 'default'}）· {stats}{tip}"
    except Exception as e:
        return "", f"优化失败：{e}"


def do_compare(label: str, original: str, optimized: str, test_input: str):
    mode = _mode_key(label)
    if not (original or "").strip():
        return "请先填写原始提示词", "", "—"
    if not (optimized or "").strip():
        return "请先完成优化", "", "—"
    if not (test_input or "").strip():
        return "请填写测试输入", "", "—"
    try:
        left, right = compare_test(original, optimized, test_input, mode)  # type: ignore[arg-type]
        stats = format_length_stats(original, optimized)
        note = (
            f"试跑完成 · 提示词{stats} · "
            f"输出字数 原文 {len(left)} / 优化后 {len(right)}"
        )
        return left, right, note
    except Exception as e:
        err = f"试跑失败：{e}"
        return err, err, err


def apply_to_report(optimized: str, label: str):
    mode = _mode_key(label)
    text = (optimized or "").strip()
    if not text:
        return gr.update(), gr.update(), "请先得到优化结果"
    if mode != "user":
        return gr.update(), gr.update(), "仅「用户提示词」模式可应用到周报任务框"
    hint = analyze_user_input(text)
    return text, hint, f"已写入周报任务框 · {hint} · 可切到「周报生成」"


def save_opt_result(optimized: str, label: str):
    text = (optimized or "").strip()
    if not text:
        return "没有可保存的优化结果"
    try:
        rel = save_optimized_prompt(text, mode=_mode_key(label))
        return f"已保存 `{rel}`"
    except Exception as e:
        return f"保存失败：{e}"


def optimize_then_report(label: str, template_choice: str, prompt: str, requirements: str, max_retry: int):
    mode = _mode_key(label)
    if mode != "user":
        msg = "「优化并生成周报」仅支持用户提示词模式"
        return "", msg, gr.update(), gr.update(), "*未生成*", "等待切换模式", "", None
    tid = parse_template_choice(template_choice) or None
    try:
        optimized = run_optimize(prompt, "user", tid, requirements)
    except Exception as e:
        return "", f"优化失败：{e}", gr.update(), gr.update(), "*未生成*", "优化失败", "", None

    stats = format_length_stats(prompt, optimized)
    hint = analyze_user_input(optimized)
    report_md, status, save_hint, download = generate_report(optimized, max_retry)
    status_line = f"已优化（{stats}）并进入周报流程\n\n{status}"
    return optimized, status_line, optimized, hint, report_md, status, save_hint, download


def build_ui() -> gr.Blocks:
    title = get_display_title()
    theme = gr.themes.Soft(primary_hue="slate")

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n{get_subtitle()}\n\nLLM：{get_llm_status()}")

        with gr.Tabs():
            with gr.Tab("周报生成"):
                with gr.Accordion("Skill 列表", open=False):
                    gr.Markdown(SKILLS_MD)

                with gr.Accordion("需求写法", open=False):
                    gr.Markdown(WRITING_TIPS_MD)

                with gr.Accordion("使用说明", open=False):
                    gr.Markdown(GUIDE_MD)

                with gr.Row():
                    with gr.Column(scale=1):
                        user_input = gr.Textbox(
                            label="分析任务",
                            placeholder=EXAMPLE_FMCG[:120] + "…",
                            lines=8,
                        )
                        input_hint = gr.Markdown(
                            value=analyze_user_input(""),
                            elem_classes=["input-hint"],
                        )

                        with gr.Row():
                            btn_template = gr.Button("填空模板", size="sm")
                            btn_ex_simple = gr.Button("简单示例", size="sm")
                            btn_ex_fmcg = gr.Button("完整示例", size="sm")
                            btn_ex_channel = gr.Button("渠道对比", size="sm")

                        max_retry = gr.Slider(0, 5, value=2, step=1, label="质量重试次数")
                        btn_generate = gr.Button("生成周报", variant="primary")

                    with gr.Column(scale=1):
                        run_status = gr.Markdown(
                            "选择示例或填写任务，看到 `[就绪]` 后点击生成。"
                        )
                        save_hint = gr.Markdown("")
                        download = gr.File(label="下载周报", interactive=False)
                        gr.Markdown(REPORTS_FOLDER_HINT)

                gr.Markdown("### 周报正文")
                report = gr.Markdown(value="*生成后显示在这里*")

                with gr.Accordion("常见问题", open=False):
                    gr.Markdown(FAQ_MD)

                user_input.change(analyze_user_input, inputs=user_input, outputs=input_hint)

                btn_template.click(lambda: EXAMPLE_TEMPLATE, outputs=user_input)
                btn_ex_simple.click(lambda: EXAMPLE_SIMPLE, outputs=user_input)
                btn_ex_fmcg.click(lambda: EXAMPLE_FMCG, outputs=user_input)
                btn_ex_channel.click(lambda: EXAMPLE_CHANNEL, outputs=user_input)

                for btn in (btn_template, btn_ex_simple, btn_ex_fmcg, btn_ex_channel):
                    btn.click(analyze_user_input, inputs=user_input, outputs=input_hint)

                btn_generate.click(
                    generate_report,
                    inputs=[user_input, max_retry],
                    outputs=[report, run_status, save_hint, download],
                )

            with gr.Tab("提示词优化"):
                gr.Markdown(OPTIMIZER_GUIDE_MD)

                with gr.Row():
                    opt_mode = gr.Radio(
                        choices=list(MODE_LABELS.keys()),
                        value="用户提示词",
                        label="优化模式",
                    )
                    opt_template = gr.Dropdown(
                        choices=templates_for_mode("user"),
                        value=default_template_choice("user"),
                        label="优化模板",
                    )

                with gr.Row():
                    with gr.Column(scale=1):
                        opt_prompt = gr.Textbox(
                            label="原始提示词",
                            lines=12,
                            placeholder="粘贴要优化的系统/用户提示词…",
                        )
                        opt_requirements = gr.Textbox(
                            label="改进需求（迭代模式）",
                            lines=3,
                            placeholder="例如：输出改为 JSON；语气更专业；不要多轮追问",
                            visible=False,
                        )
                        with gr.Row():
                            btn_opt_example = gr.Button("填入 Agent 示例", size="sm")
                            btn_optimize = gr.Button("一键优化", variant="primary")
                        with gr.Row():
                            btn_apply = gr.Button("应用到周报任务框", size="sm")
                            btn_opt_then_run = gr.Button("优化并生成周报", size="sm")
                            btn_save_opt = gr.Button("保存结果", size="sm")
                        opt_status = gr.Markdown(
                            "默认用户模式。模糊需求可先优化再跑周报；系统模式结果需手动贴回代码。"
                        )

                    with gr.Column(scale=1):
                        opt_result = gr.Textbox(label="优化结果", lines=14)

                gr.Markdown("### 试跑对比")
                test_input = gr.Textbox(
                    label="测试输入",
                    lines=3,
                    value=default_test_probe("user"),
                )
                compare_note = gr.Markdown("")
                btn_compare = gr.Button("试跑对比", variant="secondary")
                with gr.Row():
                    test_original = gr.Textbox(label="原文效果", lines=10)
                    test_optimized = gr.Textbox(label="优化后效果", lines=10)

                opt_mode.change(
                    on_mode_change,
                    inputs=opt_mode,
                    outputs=[opt_template, opt_requirements, test_input],
                )
                btn_opt_example.click(
                    fill_agent_example,
                    inputs=opt_mode,
                    outputs=[opt_prompt, opt_requirements, test_input, opt_status],
                )
                btn_optimize.click(
                    do_optimize,
                    inputs=[opt_mode, opt_template, opt_prompt, opt_requirements],
                    outputs=[opt_result, opt_status],
                )
                btn_compare.click(
                    do_compare,
                    inputs=[opt_mode, opt_prompt, opt_result, test_input],
                    outputs=[test_original, test_optimized, compare_note],
                )
                btn_apply.click(
                    apply_to_report,
                    inputs=[opt_result, opt_mode],
                    outputs=[user_input, input_hint, opt_status],
                )
                btn_save_opt.click(
                    save_opt_result,
                    inputs=[opt_result, opt_mode],
                    outputs=[opt_status],
                )
                btn_opt_then_run.click(
                    optimize_then_report,
                    inputs=[opt_mode, opt_template, opt_prompt, opt_requirements, max_retry],
                    outputs=[
                        opt_result,
                        opt_status,
                        user_input,
                        input_hint,
                        report,
                        run_status,
                        save_hint,
                        download,
                    ],
                )

    demo.theme = theme
    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        show_error=True,
    )
