"""流程执行指标统计。"""
from __future__ import annotations

from state import AgentState, RunMetrics, TaskResult


def _as_result(raw: TaskResult | dict | None, task_id: str) -> TaskResult:
    if raw is None:
        return TaskResult(task_id=task_id, success=False)
    if isinstance(raw, TaskResult):
        return raw
    return TaskResult.model_validate(raw)


def snapshot_pass_rate(state: AgentState) -> float:
    """当前任务成功率（0-100）。"""
    tasks = state.get("task_list") or []
    results = state.get("task_results") or {}
    if not tasks:
        return 0.0
    ok = sum(
        1 for t in tasks if _as_result(results.get(t.task_id), t.task_id).success
    )
    return round(ok / len(tasks) * 100, 2)


def compute_run_metrics(state: AgentState, metrics: RunMetrics) -> RunMetrics:
    """在流程结束时计算重试提升指标。"""
    tasks = state.get("task_list") or []
    results = state.get("task_results") or {}
    total = len(tasks)
    if total == 0:
        return metrics

    final_ok = sum(
        1 for t in tasks if _as_result(results.get(t.task_id), t.task_id).success
    )
    metrics.final_pass_rate = round(final_ok / total * 100, 2)

    flags = state.get("execution_flags")
    retry_count = flags.retry_count if flags else 0
    if retry_count > 0 and metrics.first_pass_rate > 0:
        metrics.improvement_pct = round(
            metrics.final_pass_rate - metrics.first_pass_rate, 2
        )
    elif retry_count == 0:
        metrics.first_pass_rate = metrics.final_pass_rate
        metrics.improvement_pct = 0.0

    return metrics


def format_metrics_report(metrics: RunMetrics) -> str:
    total_tokens = sum(c.total_tokens for c in metrics.llm_calls)
    total_latency = sum(c.latency_ms for c in metrics.llm_calls)
    lines = [
        "## 执行统计",
        f"- 首轮通过率：**{metrics.first_pass_rate}%**",
        f"- 最终通过率：**{metrics.final_pass_rate}%**",
        f"- 重试后提升：**{metrics.improvement_pct:+.2f}%**",
        f"- LLM 调用次数：**{len(metrics.llm_calls)}**",
        f"- Token 合计：**{total_tokens}**",
        f"- LLM 总耗时：**{total_latency:.0f} ms**",
        f"- 工具调用次数：**{metrics.tool_calls_count}**",
        f"- RAG 检索次数：**{metrics.rag_queries}**",
        f"- Skill 调用：**{len(metrics.skill_invocations)}** 次（{', '.join(metrics.skill_invocations) or '无'}）",
        f"- 预估节省人工：**约 {metrics.manual_minutes_saved} 分钟**",
    ]
    if metrics.llm_calls:
        lines.append("")
        lines.append("### LLM 调用明细")
        for call in metrics.llm_calls:
            lines.append(
                f"- `{call.node}` | {call.model} | {call.latency_ms}ms | "
                f"tokens={call.total_tokens}"
            )
    return "\n".join(lines)
