from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from langchain_core.messages import HumanMessage, SystemMessage

from llm_utils import get_llm, tracked_invoke
from paths import PROJECT_ROOT, TOOL_OUTPUT_DIR, ensure_user_dirs

Mode = Literal["system", "user", "iterate"]

AGENT_EXAMPLE_EXECUTOR = (
    "你是快消饮料行业任务执行专家。仅完成当前子任务，输出简洁执行报告。"
    "必须优先依据【业务文档检索结果】与工具返回的数据，不要编造与公司规则冲突的内容。"
    "数据分析任务可调用：query_sales_sql、analyze_sales_pandas、integrate_data_sources。"
    "不要替其他子任务写内容，不要写总方案汇总。"
)

AGENT_EXAMPLE_REFLECTION = (
    "质量审查员。存在失败任务时 passed 应为 false。"
    "若所有子任务 success 且实际子任务数 >= 期望条目数，passed 应为 true，score >= 95。"
)

AGENT_EXAMPLE_USER = (
    "帮我做个渠道分析，看看销量怎么样，再写个报告。"
)

# 试跑默认探针，按模式分
TEST_PROBE_SYSTEM = "当前子任务：统计便利店/商超/电商/特通渠道销售额对比。请按角色要求给出执行报告提纲（不要编造具体数字）。"
TEST_PROBE_USER = "补充：区域是华东，品类偏气泡水与电解质水，报告要有关键发现和建议动作。"
TEST_PROBE_ITERATE = "请按提示词要求，对下面执行结果做质检：任务1成功、任务2失败(SQL超时)、任务3成功。"

SYSTEM_AGENT_SYSTEM = """你在改写「Agent 系统提示词」，只输出改写后的正文，不要解释。

场景是快消饮料渠道分析 Agent，常见工具：
query_sales_sql / analyze_sales_pandas / integrate_data_sources /
query_channel_inventory / simple_web_search / RAG 业务文档。

改写要求：
- 保留原角色与硬约束，可补全工具使用边界、禁止编造、只做当前子任务
- 控制在 8～15 行中文短句，别搞 Role/Profile 长模板
- 有 {{variable}} 必须原样保留
- 不要代码块"""

SYSTEM_COMPACT_SYSTEM = """压缩系统提示词。只输出压缩后的正文。

规则：
- 保留角色、硬约束、工具名、输出格式
- 删套话、重复、空泛形容词
- 目标长度约为原文 50%～70%，能短则短
- {{variable}} 原样保留，不要代码块与解释"""

SYSTEM_GENERAL_SYSTEM = """你是提示词优化助手，专门优化「系统提示词」。
把原文改成可直接用作 system message 的版本，只输出正文。

结构按需覆盖：Role / Skills / Rules / Workflow（别强行堆满空章节）。
保留意图与关键约束；{{variable}} 原样保留；不要代码块与解释。"""

USER_FMCG_SYSTEM = """你在优化「快消渠道周报」的用户任务描述。只输出优化后的任务正文。

目标：让下游 Agent 能稳定拆成 3～6 条子任务，并更容易命中 Skill（渠道对比/SKU排行/趋势/政策/整合周报）。

写法：
- 用中文编号列表 1. 2. 3. …
- 每条只做一件事，带业务词：渠道/SKU/政策/库存/竞品/周报
- 可补区域或品类，但不要编造用户没提的具体品牌 KPI
- 不要回答任务，不要解释，不要代码块"""

USER_BASIC_SYSTEM = """优化用户提示词本身，使其更清晰可执行；不是去回答它。
直接输出优化后的用户提示词。
消除模糊词；明确目标、对象、约束、输出格式；不添加用户未提的新需求；{{variable}} 原样保留。"""

USER_COMPACT_SYSTEM = """压缩用户提示词。只输出压缩后正文。
保留任务目标与关键约束，删废话；尽量短于原文；{{variable}} 原样保留；不要解释。"""

ITERATE_SYSTEM = """按改进需求微调已有提示词，只输出修改后的完整提示词。
保持原语言与结构，精准改、别过度扩写；{{variable}} 原样保留；不要代码块与解释。"""

ITERATE_SHORTEN_SYSTEM = """用户要「变短」。在保留核心约束与可执行性的前提下压缩提示词。
只输出压缩后的完整提示词；能合并的合并；不要解释。"""

TEMPLATES: dict[str, dict] = {
    "system_agent": {
        "id": "system_agent",
        "name": "Agent短约束",
        "mode": "system",
        "description": "贴合本项目工具边界，短句版 system",
        "system": SYSTEM_AGENT_SYSTEM,
    },
    "system_compact": {
        "id": "system_compact",
        "name": "系统精简",
        "mode": "system",
        "description": "压缩长度，适合控 token",
        "system": SYSTEM_COMPACT_SYSTEM,
    },
    "system_general": {
        "id": "system_general",
        "name": "通用结构化",
        "mode": "system",
        "description": "Role/Skills/Rules/Workflow 重组",
        "system": SYSTEM_GENERAL_SYSTEM,
    },
    "user_fmcg": {
        "id": "user_fmcg",
        "name": "周报任务",
        "mode": "user",
        "description": "改成编号任务列表，便于拆分与 Skill",
        "system": USER_FMCG_SYSTEM,
    },
    "user_basic": {
        "id": "user_basic",
        "name": "用户基础优化",
        "mode": "user",
        "description": "消除模糊，补关键约束",
        "system": USER_BASIC_SYSTEM,
    },
    "user_compact": {
        "id": "user_compact",
        "name": "用户精简",
        "mode": "user",
        "description": "压短用户提示词",
        "system": USER_COMPACT_SYSTEM,
    },
    "iterate": {
        "id": "iterate",
        "name": "通用迭代",
        "mode": "iterate",
        "description": "按改进需求微调",
        "system": ITERATE_SYSTEM,
    },
    "iterate_shorten": {
        "id": "iterate_shorten",
        "name": "迭代压缩",
        "mode": "iterate",
        "description": "按需求改，并尽量变短",
        "system": ITERATE_SHORTEN_SYSTEM,
    },
}


def list_templates(mode: Optional[Mode] = None) -> list[dict]:
    items = []
    for t in TEMPLATES.values():
        if mode is None or t["mode"] == mode:
            items.append(
                {
                    "id": t["id"],
                    "name": t["name"],
                    "mode": t["mode"],
                    "description": t["description"],
                }
            )
    return items


def templates_for_mode(mode: Mode) -> list[str]:
    return [f"{t['id']} · {t['name']}" for t in list_templates(mode)]


def default_template_choice(mode: Mode) -> str:
    choices = templates_for_mode(mode)
    return choices[0] if choices else ""


def parse_template_choice(choice: str) -> str:
    raw = (choice or "").strip()
    if not raw:
        return ""
    return raw.split("·", 1)[0].strip()


def default_test_probe(mode: Mode) -> str:
    if mode == "user":
        return TEST_PROBE_USER
    if mode == "iterate":
        return TEST_PROBE_ITERATE
    return TEST_PROBE_SYSTEM


def length_stats(original: str, optimized: str) -> dict:
    a = len((original or "").strip())
    b = len((optimized or "").strip())
    delta = b - a
    pct = round(delta / a * 100, 1) if a else 0.0
    return {
        "original_chars": a,
        "optimized_chars": b,
        "delta_chars": delta,
        "delta_pct": pct,
    }


def format_length_stats(original: str, optimized: str) -> str:
    s = length_stats(original, optimized)
    sign = "+" if s["delta_chars"] >= 0 else ""
    return (
        f"长度 {s['original_chars']} → {s['optimized_chars']} 字"
        f"（{sign}{s['delta_chars']}，{sign}{s['delta_pct']}%）"
    )


def save_optimized_prompt(text: str, mode: str = "system", tag: str = "") -> str:
    ensure_user_dirs()
    body = (text or "").strip()
    if not body:
        raise ValueError("没有可保存的内容")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_mode = (mode or "prompt").strip() or "prompt"
    suffix = f"_{tag}" if tag else ""
    fp = TOOL_OUTPUT_DIR / f"prompt_{safe_mode}{suffix}_{ts}.md"
    fp.write_text(body, encoding="utf-8")
    try:
        rel = str(fp.relative_to(PROJECT_ROOT))
    except ValueError:
        rel = str(fp)
    print(f"[prompt_opt] saved {rel}")
    return rel


def _need_llm():
    llm = get_llm()
    if llm is None:
        raise RuntimeError("未配置 API Key，请在 .env 中设置 DEEPSEEK_API_KEY 或 OPENAI_API_KEY")
    return llm


def _clean_output(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def _invoke(system: str, user: str, node: str) -> str:
    llm = _need_llm()
    reply = tracked_invoke(
        llm,
        [SystemMessage(content=system), HumanMessage(content=user)],
        node=node,
    )
    content = reply.content if isinstance(reply.content, str) else str(reply.content)
    return _clean_output(content)


def optimize_prompt(
    prompt: str,
    mode: Mode = "system",
    template_id: Optional[str] = None,
) -> str:
    text = (prompt or "").strip()
    if not text:
        raise ValueError("prompt 不能为空")
    if mode not in ("system", "user"):
        raise ValueError("optimize_prompt 仅支持 mode=system|user，迭代请用 iterate_prompt")

    tid = template_id or ("system_agent" if mode == "system" else "user_fmcg")
    tpl = TEMPLATES.get(tid)
    if tpl is None or tpl["mode"] != mode:
        raise ValueError(f"模板不可用: {tid}（mode={mode}）")

    if mode == "system":
        user = f"请优化以下系统提示词，直接输出优化后的完整提示词：\n\n{text}"
        node = "prompt_opt_system"
    else:
        user = f"请优化以下用户提示词，直接输出优化后的提示词：\n\n{text}"
        node = "prompt_opt_user"

    print(f"[prompt_opt] mode={mode} tpl={tid} len={len(text)}")
    return _invoke(tpl["system"], user, node)


def iterate_prompt(prompt: str, requirements: str, template_id: Optional[str] = None) -> str:
    text = (prompt or "").strip()
    req = (requirements or "").strip()
    if not text:
        raise ValueError("prompt 不能为空")
    if not req:
        raise ValueError("requirements 不能为空")

    tid = template_id or "iterate"
    tpl = TEMPLATES.get(tid)
    if tpl is None or tpl["mode"] != "iterate":
        raise ValueError(f"迭代模板不可用: {tid}")

    user = (
        "请根据改进需求修改原始提示词，直接输出修改后的完整提示词。\n\n"
        f"【原始提示词】\n{text}\n\n"
        f"【改进需求】\n{req}"
    )
    print(f"[prompt_opt] iterate tpl={tid} len={len(text)} req_len={len(req)}")
    return _invoke(tpl["system"], user, "prompt_opt_iterate")


def run_optimize(
    prompt: str,
    mode: Mode,
    template_id: Optional[str] = None,
    requirements: str = "",
) -> str:
    if mode == "iterate":
        return iterate_prompt(prompt, requirements, template_id)
    return optimize_prompt(prompt, mode, template_id)


def test_prompt(prompt: str, test_input: str, mode: Mode = "system") -> str:
    text = (prompt or "").strip()
    probe = (test_input or "").strip()
    if not text:
        raise ValueError("prompt 不能为空")
    if not probe:
        raise ValueError("test_input 不能为空")

    llm = _need_llm()
    if mode == "user":
        messages = [HumanMessage(content=f"{text}\n\n---\n补充输入：\n{probe}")]
    else:
        messages = [SystemMessage(content=text), HumanMessage(content=probe)]

    print(f"[prompt_opt] test mode={mode} prompt_len={len(text)} input_len={len(probe)}")
    reply = tracked_invoke(llm, messages, node="prompt_opt_test")
    content = reply.content if isinstance(reply.content, str) else str(reply.content)
    return _clean_output(content)


def compare_test(
    original: str,
    optimized: str,
    test_input: str,
    mode: Mode = "system",
) -> tuple[str, str]:
    left = test_prompt(original, test_input, mode)
    right = test_prompt(optimized, test_input, mode)
    return left, right


def example_for_mode(mode: Mode) -> tuple[str, str]:
    if mode == "system":
        return AGENT_EXAMPLE_EXECUTOR, ""
    if mode == "user":
        return AGENT_EXAMPLE_USER, ""
    return AGENT_EXAMPLE_REFLECTION, "要求输出 JSON，字段含 passed/score/issues/suggestions/summary，语气更严谨，整体再短一点。"
