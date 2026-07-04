"""Gradio 网页：面向非技术用户的使用引导 + 报告展示与下载。"""
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
from tools import count_expected_tasks

# ── 示例需求（点击即可填入） ──────────────────────────────────────────
EXAMPLE_SIMPLE = "撰写产品发布说明、通知团队成员、整理上线检查清单"

EXAMPLE_618 = (
    "策划电商618大促完整运营方案，需要完成全部子任务："
    "1. 制定满减、优惠券、跨店折扣活动规则；"
    "2. 设计首页会场、商品分会场页面布局与文案；"
    "3. 撰写直播间带货脚本、主播话术、活动福利介绍；"
    "4. 规划短视频宣传内容、投放平台与发布排期；"
    "5. 整理客服常见活动问答、售后赔付规则；"
    "6. 预估活动流量、备货库存、预算成本核算；"
    "7. 活动结束后复盘数据指标与优化方向。"
)

EXAMPLE_CS = "整理618活动客服常见问答、售后赔付规则说明、投诉处理话术"

EXAMPLE_TEMPLATE = """我要【填写背景，如：策划618大促 / 新品上线】，需要完成：
1. 【第一项，如：制定满减和优惠券规则】
2. 【第二项，如：设计活动页面布局与文案】
3. 【第三项，如：撰写直播带货脚本】
（可按需继续写 4、5、6…，建议一次不超过 10 项）"""

WRITING_TIPS_MD = """
### ✍️ 怎样写，AI 更容易理解？

**您不需要会「提示词」**，只要像给同事布置工作一样，**把要做的事列清楚**即可。

---

#### ✅ 推荐写法（三选一）

| 写法 | 示例 |
|------|------|
| **用顿号/逗号列多项** | 撰写发布说明、通知团队、整理检查清单 |
| **用编号列多项** | 1. 制定活动规则；2. 设计页面；3. 写直播脚本 |
| **先写背景 + 再列任务** | 我要策划618大促，需要：制定规则、设计页面、写脚本… |

---

#### ❌ 尽量避免

| 不推荐 | 为什么 | 改成 |
|--------|--------|------|
| 帮我做个方案 | 太笼统，不知道做哪几块 | 列出 3～7 件具体要做的事 |
| 越详细越好 | 没有具体任务 | 写「撰写FAQ、整理赔付规则」 |
| 一次写 20 件事 | 超出系统上限（10 项） | 拆成两次生成，或只保留最重要的 10 项 |

---

#### 💡 写好需求的 4 个小技巧

1. **一条一事**：每条任务说一件事，如「写直播脚本」，不要多条揉在一起  
2. **带一点背景**：开头写「618大促 / 新品发布」，方案更贴场景  
3. **3～7 项最合适**：太少内容薄，太多等待久  
4. **不会写就点「填空模板」或示例按钮**，改几个字就能用  

> 输入框下方会**实时提示**系统能否理解您的写法，请留意绿色 ✅ 或黄色 ⚠️ 提示。
"""

GUIDE_MD = """
### 📋 使用步骤（无需编程）

1. **看书写建议**：展开上方「如何写好需求」，或点「填空模板 / 示例」
2. **写需求**：在输入框描述工作，看到 ✅ 提示后再点生成
3. **点生成**：点击「生成方案」，等待 **3～8 分钟**
4. **看结果 & 下载**：下方阅读报告，右侧下载 Markdown 文件

> 💡 系统会自动参考公司业务文档（如 618 规则），让输出更贴合实际。
"""

FAQ_MD = """
**Q：需要会编程或会写 AI 提示词吗？**  
不需要。用日常中文列任务即可。

**Q：输入框下面的提示是什么意思？**  
✅ 表示系统已识别多项任务，可以生成；⚠️ 表示描述偏笼统，建议按书写指南改一改。

**Q：等很久正常吗？**  
正常。7 项任务大约 3～8 分钟，请勿重复点击。

**Q：报告保存在哪？**  
自动保存到 **`reports`** 文件夹，也可网页右侧下载。

**Q：运行失败怎么办？**  
检查 API Key；或将「质量检查重试」调为 0 后重试。
"""

REPORTS_FOLDER_HINT = (
    f"📁 报告默认保存文件夹：\n`{REPORTS_DIR.resolve()}`\n\n"
    "用资源管理器打开上述路径，即可查看所有历史报告。"
)


def analyze_user_input(text: str) -> str:
    """实时分析用户输入，给出友好书写提示。"""
    text = (text or "").strip()
    if not text:
        return (
            "💡 **请输入需求**，或点击「填空模板 / 示例按钮」。\n\n"
            "推荐格式：`背景 + 用顿号或编号列出 3～7 项具体工作`"
        )

    if "【" in text and "】" in text:
        return (
            "📝 **您正在使用填空模板**：请把【】里的占位文字改成您的真实内容，"
            "改完后再点「生成方案」。"
        )

    if len(text) < 8:
        return "⚠️ **内容太短**：请补充具体任务，例如「写发布说明、通知团队、整理清单」。"

    vague_only = re.fullmatch(
        r"[\s\d]*(?:帮我|请帮|做一个|弄个|搞个|整一个|生成|写个)?"
        r"(?:方案|计划|报告|文档|东西)?[\s\d]*",
        text,
    )
    if vague_only or (len(text) < 20 and count_expected_tasks(text) <= 1):
        return (
            "⚠️ **描述偏笼统**：AI 不清楚具体要做哪几件事。\n\n"
            "建议改成：\n"
            "- `撰写XX、设计XX、整理XX`（用顿号分开）\n"
            "- 或 `1. …；2. …；3. …`（用编号分开）"
        )

    n = count_expected_tasks(text)
    has_list_markers = bool(
        re.search(r"\d+[\.、]\s*", text) or "、" in text or "，" in text or ";" in text
    )

    if n > 10:
        return (
            f"⚠️ **任务过多（约 {n} 项）**：系统最多处理 **10** 项。\n\n"
            "请删减次要任务，或分两次生成。"
        )

    if n >= 3:
        return (
            f"✅ **写法很好！** 预计拆分为 **{n}** 个子任务。\n\n"
            "确认内容无误后，点击「生成方案」即可。"
        )

    if n == 2:
        return (
            f"✅ **可以生成。** 预计 **{n}** 个子任务。\n\n"
            "如需更完整方案，可再补充 1～2 项具体工作。"
        )

    if n == 1 and has_list_markers:
        return (
            "⚠️ **只识别到 1 项任务**：您用了分隔符，但可能格式不标准。\n\n"
            "建议用：`任务A、任务B、任务C` 或 `1. 任务A；2. 任务B`"
        )

    if n == 1:
        return (
            "⚠️ **目前只有 1 项任务**。\n\n"
            "若实际有多件事，请用 **顿号「、」** 或 **编号 1. 2. 3.** 分开写，"
            "例如：`制定规则、设计页面、写直播脚本`"
        )

    return "💡 请按书写指南列出具休任务后再生成。"


def generate_plan(user_input: str, max_retry: int) -> tuple[str, str, str, str | None]:
    """生成方案并返回：报告、友好状态、保存说明、可下载文件路径。"""
    if not user_input.strip():
        return (
            "⚠️ 请先在上方输入您的需求，或点击示例 / 填空模板。",
            "等待输入…",
            "",
            None,
        )

    hint = analyze_user_input(user_input)
    if hint.startswith("⚠️") and "模板" not in user_input:
        return (
            f"**请先完善需求描述**\n\n{hint}",
            "请先根据提示修改需求，再点击生成。",
            "",
            None,
        )

    if "【" in user_input and "】" in user_input:
        return (
            "**请先完成填空模板**\n\n"
            "您还有【】占位符未修改，请改成真实内容后再生成。",
            "等待您完成模板填写…",
            "",
            None,
        )

    try:
        state = run(user_input.strip(), verbose=False, max_retry=int(max_retry))
        final_output = state.get("final_output") or "（未生成内容，请重试）"
        thread_id = state.get("thread_id", "")
        saved_path = save_user_report(final_output, thread_id)

        metrics = state.get("run_metrics")
        m = metrics.model_dump() if hasattr(metrics, "model_dump") else {}

        status = (
            "✅ **生成完成！**\n\n"
            f"- 子任务数：{len(state.get('task_list') or [])}\n"
            f"- 最终通过率：{m.get('final_pass_rate', 0)}%\n"
            f"- 耗时统计：LLM 调用 {len(m.get('llm_calls', []))} 次\n\n"
            "请在下方阅读报告，或点击右侧 **下载报告** 保存到电脑。"
        )

        save_hint = (
            f"✅ 已自动保存\n\n"
            f"**文件名**：`{saved_path.name}`\n\n"
            f"**文件夹**：\n`{saved_path.parent}`\n\n"
            "您可以用记事本、Word 或 Typora 打开 `.md` 文件查看。"
        )

        return final_output, status, save_hint, str(saved_path)

    except Exception as e:
        err_status = (
            f"❌ 生成失败：{e}\n\n"
            "建议：\n"
            "1. 检查网络是否正常\n"
            "2. 确认 `.env` 中已配置 DeepSeek API Key\n"
            "3. 将「质量检查重试」调为 0 后重试\n"
            "4. 若仍失败，请联系技术人员查看终端日志"
        )
        return f"**运行失败**\n\n{e}", err_status, "", None


def build_ui() -> gr.Blocks:
    theme = gr.themes.Soft(primary_hue="orange")
    with gr.Blocks(
        title="AI 任务方案生成器",
        theme=theme,
        css=(
            ".guide-box { padding: 12px; border-radius: 8px; background: #fff8f0; }"
            ".input-hint { padding: 10px; border-radius: 6px; border: 1px solid #ffd591; "
            "background: #fffbe6; min-height: 60px; }"
        ),
    ) as demo:
        gr.Markdown(
            f"# 🚀 AI 任务方案生成器\n"
            f"用**日常中文**列出要做的事即可，无需会编程或写 AI 提示词。\n\n"
            f"当前模型：**{get_llm_status()}**"
        )

        with gr.Accordion("✍️ 如何写好需求？（推荐先看，不用会提示词）", open=True):
            gr.Markdown(WRITING_TIPS_MD)

        with gr.Accordion("📖 使用步骤", open=False):
            gr.Markdown(GUIDE_MD)

        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 第一步：描述您的需求")
                user_input = gr.Textbox(
                    label="请输入要完成的工作",
                    placeholder=(
                        "推荐格式：\n"
                        "我要策划618大促，需要完成：\n"
                        "1. 制定满减规则；2. 设计页面；3. 写直播脚本\n\n"
                        "或：撰写发布说明、通知团队、整理检查清单"
                    ),
                    lines=8,
                )
                input_hint = gr.Markdown(
                    value=analyze_user_input(""),
                    elem_classes=["input-hint"],
                    label="书写提示（实时）",
                )

                gr.Markdown("**快捷填入（点按钮即可，再按提示修改）：**")
                with gr.Row():
                    btn_template = gr.Button("📝 填空模板", size="sm")
                    btn_ex_simple = gr.Button("📌 简单3项", size="sm")
                    btn_ex_618 = gr.Button("📌 618大促", size="sm")
                    btn_ex_cs = gr.Button("📌 客服问答", size="sm")

                gr.Markdown("### 第二步：点击生成")
                max_retry = gr.Slider(
                    0,
                    5,
                    value=2,
                    step=1,
                    label="质量检查重试次数",
                    info="不懂请保持默认 2",
                )
                btn_generate = gr.Button("🎯 生成方案", variant="primary", size="lg")

            with gr.Column(scale=1):
                gr.Markdown("### 第三步：查看与下载")
                run_status = gr.Markdown(
                    "👋 欢迎！请先在左侧输入或选择示例，**看到绿色 ✅ 提示**后再点生成。"
                )
                save_hint = gr.Markdown("")
                download = gr.File(
                    label="⬇️ 下载报告（Markdown 文件）",
                    interactive=False,
                )
                gr.Markdown(REPORTS_FOLDER_HINT)

        gr.Markdown("---")
        gr.Markdown("### 📄 方案报告（生成后显示在下方）")
        report = gr.Markdown(value="*报告将显示在这里…*")

        with gr.Accordion("❓ 常见问题", open=False):
            gr.Markdown(FAQ_MD)

        # 实时书写提示
        user_input.change(analyze_user_input, inputs=user_input, outputs=input_hint)

        # 示例 / 模板按钮
        btn_template.click(lambda: EXAMPLE_TEMPLATE, outputs=user_input)
        btn_ex_simple.click(lambda: EXAMPLE_SIMPLE, outputs=user_input)
        btn_ex_618.click(lambda: EXAMPLE_618, outputs=user_input)
        btn_ex_cs.click(lambda: EXAMPLE_CS, outputs=user_input)

        # 填入示例后也刷新提示
        for btn in (btn_template, btn_ex_simple, btn_ex_618, btn_ex_cs):
            btn.click(analyze_user_input, inputs=user_input, outputs=input_hint)

        btn_generate.click(
            generate_plan,
            inputs=[user_input, max_retry],
            outputs=[report, run_status, save_hint, download],
        )

    return demo


if __name__ == "__main__":
    build_ui().launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("GRADIO_PORT", "7860")),
        show_error=True,
    )
