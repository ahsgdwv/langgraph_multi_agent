"""Gradio 网页：快消业务分析入口。"""
from __future__ import annotations

import os
import re

import gradio as gr
from dotenv import load_dotenv

load_dotenv()
os.environ.setdefault("AUTO_APPROVE_HUMAN", "true")

from llm_utils import get_llm_status
from main import run
from paths import REPORTS_DIR, save_user_report
from skills import list_skills
from tools import count_expected_tasks

HINT_OK = "[就绪]"
HINT_WARN = "[注意]"
HINT_INFO = "[提示]"

EXAMPLE_FMCG = (
    "元气森林华东区渠道分析，需要完成："
    "1. 统计便利店、商超、电商、特通渠道销量与销售额对比；"
    "2. 输出气泡水及电解质水 SKU 销售 TOP 排行；"
    "3. 检索渠道陈列费与定价政策；"
    "4. 整合 SQL 销售数据、pandas 统计与政策文档生成周报；"
    "5. 查询核心 SKU 渠道库存；"
    "6. 调研竞品无糖气泡水市场动态。"
)

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
        err = (
            f"生成失败：{e}\n"
            "请检查网络与 API Key，或将质量重试次数设为 0。"
        )
        return f"运行失败\n\n{e}", err, "", None


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="slate")
    with gr.Blocks(title="元气森林渠道数据分析 Agent") as demo:
        gr.Markdown(
            "# 元气森林渠道数据分析 Agent\n"
            "渠道销量统计 · SKU 排行 · 政策检索 · 多源整合周报\n\n"
            f"LLM：{get_llm_status()}"
        )

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
                run_status = gr.Markdown("选择示例或填写任务，看到 `[就绪]` 后点击生成。")
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

    demo.theme = theme
    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        show_error=True,
    )
