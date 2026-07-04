"""多 Agent 节点：文档加载、分发、执行、研究、反思、总结。"""
from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Optional, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel

from external_tools import EXECUTOR_TOOLS, TOOL_MAP
from llm_utils import get_llm, get_llm_status, tracked_invoke
from metrics import compute_run_metrics, format_metrics_report, snapshot_pass_rate
from rag_store import format_retrieved_context, ingest_documents, retrieve_context
from state import (
    AgentState,
    ExecutionFlags,
    HumanApprovalRecord,
    ReflectionRecord,
    RunMetrics,
    SubTask,
    TaskResult,
)
from tools import apply_aggregate_to_state, apply_split_to_state, count_expected_tasks

T = TypeVar("T", bound=BaseModel)

__all__ = [
    "get_llm_status",
    "document_loader_agent_node",
    "dispatcher_agent_node",
    "human_review_dispatch_node",
    "executor_agent_node",
    "researcher_agent_node",
    "reflection_agent_node",
    "human_review_summary_node",
    "summary_agent_node",
    "AGENT_NODES",
]


def _parse_json_from_text(text: str) -> dict[str, Any]:
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        text = match.group(1).strip()
    return json.loads(text)


def _invoke_structured_json(
    llm: ChatOpenAI, model_cls: type[T], messages: list[BaseMessage], metrics: RunMetrics
) -> T:
    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    system = messages[0]
    rest = messages[1:]
    structured_messages = [
        SystemMessage(
            content=(
                f"{system.content}\n\n"
                "请仅输出一个符合以下 JSON Schema 的 JSON 对象，不要 markdown 或其他说明：\n"
                f"{schema}"
            )
        ),
        *rest,
    ]
    response = tracked_invoke(llm, structured_messages, node="reflection", metrics=metrics)
    content = response.content
    if not isinstance(content, str):
        content = str(content)
    return model_cls.model_validate(_parse_json_from_text(content))


def _merge_flags(state: AgentState, **kwargs) -> ExecutionFlags:
    current = state.get("execution_flags") or ExecutionFlags()
    if isinstance(current, dict):
        current = ExecutionFlags.model_validate(current)
    data = current.model_dump()
    data.update(kwargs)
    return ExecutionFlags(**data)


def _get_metrics(state: AgentState) -> RunMetrics:
    raw = state.get("run_metrics") or RunMetrics()
    if isinstance(raw, dict):
        return RunMetrics.model_validate(raw)
    return raw


def _as_task_result(raw: TaskResult | dict | None, task_id: str) -> TaskResult:
    if raw is None:
        return TaskResult(task_id=task_id, success=False, error="未执行")
    if isinstance(raw, TaskResult):
        return raw
    if isinstance(raw, dict):
        return TaskResult.model_validate(raw)
    return TaskResult(task_id=task_id, success=False, error="未知结果类型")


def _auto_approve_enabled() -> bool:
    return os.getenv("AUTO_APPROVE_HUMAN", "").lower() in ("1", "true", "yes")


def _is_stage_approved(state: AgentState, stage: Literal["post_dispatch", "pre_summary"]) -> bool:
    for record in state.get("human_approvals") or []:
        if record.stage == stage and record.approved:
            return True
    return False


def _process_human_decision(
    state: AgentState,
    stage: Literal["post_dispatch", "pre_summary"],
    decision: Any,
) -> dict:
    if isinstance(decision, dict):
        approved = bool(decision.get("approved", False))
        comment = str(decision.get("comment", ""))
        reviewer = str(decision.get("reviewer", "human"))
    else:
        approved = bool(decision)
        comment = "自动确认" if approved else "拒绝"
        reviewer = "human"

    record = HumanApprovalRecord(
        stage=stage, approved=approved, comment=comment, reviewer=reviewer
    )
    approvals = list(state.get("human_approvals") or [])
    approvals.append(record)
    stage_label = "分发" if stage == "post_dispatch" else "总结"
    status = "已批准" if approved else "已拒绝"
    return {
        "human_approvals": approvals,
        "execution_flags": _merge_flags(state, awaiting_human=False, pending_human_stage=None),
        "messages": [AIMessage(content=f"[人工确认/{stage_label}] {status} {comment or ''}")],
    }


def document_loader_agent_node(state: AgentState) -> dict:
    """加载业务文档到 Chroma 向量库。"""
    flags = state.get("execution_flags") or ExecutionFlags()
    if flags.documents_loaded:
        return {"messages": [AIMessage(content="[文档加载] 已加载，跳过")]}

    result = ingest_documents()
    return {
        "execution_flags": _merge_flags(state, documents_loaded=True),
        "messages": [AIMessage(content=f"[文档加载] {result['message']}")],
    }


def dispatcher_agent_node(state: AgentState) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    if flags.planning_done and state.get("task_list"):
        return {"messages": [AIMessage(content="[分发] 已拆分，跳过")]}

    patch = apply_split_to_state(state["user_input"])
    task_list: list[SubTask] = patch["task_list"]
    new_flags = _merge_flags(state, **patch["execution_flags"].model_dump())
    return {
        "task_list": task_list,
        "execution_flags": new_flags,
        "messages": [
            AIMessage(
                content=f"[分发] {len(task_list)} 项："
                + "、".join(t.title for t in task_list)
            )
        ],
    }


def human_review_dispatch_node(state: AgentState) -> dict:
    if _is_stage_approved(state, "post_dispatch"):
        return {"messages": [AIMessage(content="[人工确认] 分发已确认")]}

    if _auto_approve_enabled():
        return _process_human_decision(
            state, "post_dispatch", {"approved": True, "comment": "自动批准"}
        )

    task_list = state.get("task_list") or []
    decision = interrupt(
        {
            "type": "human_review",
            "stage": "post_dispatch",
            "prompt": "确认子任务拆分后继续执行",
            "task_list": [t.model_dump() for t in task_list],
            "thread_id": state.get("thread_id", ""),
        }
    )
    update = _process_human_decision(state, "post_dispatch", decision)
    update["execution_flags"] = _merge_flags(
        {**state, **update}, awaiting_human=False, pending_human_stage=None
    )
    return update


def _maybe_prefetch_tools(task: SubTask, metrics: RunMetrics) -> list[str]:
    """规则触发常用工具（无 Function Calling 时的兜底）。"""
    text = f"{task.title} {task.description}"
    outputs: list[str] = []
    if re.search(r"库存|备货", text):
        out = TOOL_MAP["query_activity_inventory"].invoke({"sku": "DEFAULT"})
        outputs.append(f"[库存查询] {out}")
        metrics.tool_calls_count += 1
    if re.search(r"搜索|竞品|调研", text):
        out = TOOL_MAP["simple_web_search"].invoke({"query": text[:80]})
        outputs.append(f"[搜索] {out}")
        metrics.tool_calls_count += 1
    return outputs


def _run_llm_with_tools(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    metrics: RunMetrics,
    node: str,
) -> tuple[str, list[str]]:
    tool_names: list[str] = []
    llm_tools = llm.bind_tools(EXECUTOR_TOOLS)
    conversation = list(messages)

    for _ in range(3):
        response = tracked_invoke(llm_tools, conversation, node=node, metrics=metrics)
        if not getattr(response, "tool_calls", None):
            return str(response.content), tool_names

        conversation.append(response)
        for call in response.tool_calls:
            name = call["name"]
            tool_names.append(name)
            metrics.tool_calls_count += 1
            fn = TOOL_MAP.get(name)
            try:
                result = fn.invoke(call["args"]) if fn else f"未知工具: {name}"
            except Exception as e:
                result = str(e)
            conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return str(response.content), tool_names


def _execute_single_task(
    task: SubTask,
    state: AgentState,
    llm: Optional[ChatOpenAI],
    *,
    rag_k: int = 4,
    node: str = "executor",
) -> TaskResult:
    fail_ids = state.get("simulate_fail_task_ids") or []
    flags = state.get("execution_flags") or ExecutionFlags()
    metrics = _get_metrics(state)

    if task.task_id in fail_ids and flags.retry_count < 1:
        return TaskResult(
            task_id=task.task_id,
            success=False,
            error=f"执行失败（{task.task_id}）",
            tool_calls=["simulate_failure"],
        )

    query = f"{task.title} {task.description} {state['user_input'][:200]}"
    docs = retrieve_context(query, k=rag_k)
    metrics.rag_queries += 1
    rag_text = format_retrieved_context(docs)
    prefetched = _maybe_prefetch_tools(task, metrics)

    if llm is not None:
        system = SystemMessage(
            content=(
                "你是任务执行专家。仅完成当前子任务，输出简洁执行报告。"
                "必须优先依据【业务文档检索结果】，不要编造与公司规则冲突的内容。"
                "不要替其他子任务写内容，不要写总方案汇总。"
                "可调用工具：文件读写、库存查询、模拟搜索。"
            )
        )
        human = HumanMessage(
            content=(
                f"总需求背景：{state['user_input']}\n"
                f"当前子任务：{task.model_dump_json(ensure_ascii=False)}\n\n"
                f"【业务文档检索结果】\n{rag_text}\n"
                + ("\n".join(prefetched) if prefetched else "")
            )
        )
        try:
            output, tool_calls = _run_llm_with_tools(
                llm, [system, human], metrics, node=node
            )
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output=str(output),
                tool_calls=tool_calls or ["llm"],
            )
        except Exception as e:
            return TaskResult(task_id=task.task_id, success=False, error=str(e))

    local_out = f"已完成：{task.title}"
    if docs:
        local_out += f"\n（参考文档：{docs[0].source}）"
    return TaskResult(
        task_id=task.task_id,
        success=True,
        output=local_out,
        tool_calls=["local_executor"],
    )


def _run_task_node(state: AgentState, *, node: str, rag_k: int) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    metrics = _get_metrics(state)
    task_list: list[SubTask] = list(state.get("task_list") or [])
    task_results: dict[str, TaskResult] = dict(state.get("task_results") or {})
    idx = flags.current_task_index

    if idx >= len(task_list):
        return {
            "execution_flags": _merge_flags(state, execution_done=True),
            "run_metrics": metrics,
            "messages": [AIMessage(content=f"[{node}] 全部完成")],
        }

    if flags.retry_count == 0 and idx == 0 and metrics.first_pass_rate == 0:
        pass  # 首轮开始前不快照

    current = task_list[idx]
    updated_list = list(task_list)
    updated_list[idx] = current.model_copy(update={"status": "running"})

    result = _execute_single_task(
        current, state, get_llm(), rag_k=rag_k, node=node
    )
    task_results[current.task_id] = result
    final_status = "success" if result.success else "failed"
    updated_list[idx] = current.model_copy(update={"status": final_status})

    next_idx = idx + 1
    all_done = next_idx >= len(task_list)
    new_flags = _merge_flags(
        state,
        current_task_index=next_idx,
        execution_done=all_done,
        need_retry=False,
    )

    if all_done and flags.retry_count == 0:
        snap_state = {**state, "task_list": updated_list, "task_results": task_results}
        metrics.first_pass_rate = snapshot_pass_rate(snap_state)  # type: ignore[arg-type]

    status = "成功" if result.success else "失败"
    retry = f" retry={flags.retry_count}/{flags.max_retry}" if flags.retry_count else ""
    detail = (result.output or result.error or "")[:120]
    retrieved = retrieve_context(f"{current.title} {current.description}", k=2)

    return {
        "task_list": updated_list,
        "task_results": task_results,
        "execution_flags": new_flags,
        "retrieved_context": retrieved,
        "run_metrics": metrics,
        "messages": [AIMessage(content=f"[{node}]{retry} {current.task_id} {status} {detail}")],
    }


def executor_agent_node(state: AgentState) -> dict:
    return _run_task_node(state, node="executor", rag_k=4)


def researcher_agent_node(state: AgentState) -> dict:
    """研究型 Agent：更深 RAG 检索，适合规则/文档类任务。"""
    return _run_task_node(state, node="researcher", rag_k=6)


def _build_reflection_rule_based(state: AgentState, round_index: int) -> ReflectionRecord:
    task_list = state.get("task_list") or []
    task_results = state.get("task_results") or {}
    flags = state.get("execution_flags") or ExecutionFlags()

    failed = [
        t for t in task_list
        if not _as_task_result(task_results.get(t.task_id), t.task_id).success
    ]
    success_count = len(task_list) - len(failed)
    expected = count_expected_tasks(state["user_input"])
    actual = len(task_list)
    coverage_ok = actual >= expected if expected else True

    score = int(success_count / max(len(task_list), 1) * 100)
    if coverage_ok and not failed:
        score = max(score, 95)
    elif not coverage_ok:
        score = min(score, 85)

    issues: list[str] = [f"{t.task_id} 失败" for t in failed]
    if not coverage_ok:
        issues.append(f"子任务拆分不足：期望约 {expected} 项，实际 {actual} 项")

    return ReflectionRecord(
        round_index=round_index,
        passed=len(failed) == 0 and coverage_ok,
        score=score,
        issues=issues,
        suggestions=["重试失败任务"] if failed else [],
        summary=(
            f"{success_count}/{len(task_list)} 成功，期望 {expected} 项，"
            f"重试 {flags.retry_count}/{flags.max_retry}"
        ),
    )


def reflection_agent_node(state: AgentState) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    metrics = _get_metrics(state)
    logs = list(state.get("reflection_logs") or [])
    round_index = len(logs) + 1

    llm = get_llm()
    if llm:
        task_list = state.get("task_list") or []
        task_results = state.get("task_results") or {}
        expected = count_expected_tasks(state["user_input"])
        system = SystemMessage(
            content=(
                "质量审查员。存在失败任务时 passed 应为 false。"
                "若所有子任务 success 且实际子任务数 >= 期望条目数，passed 应为 true，score >= 95。"
            )
        )
        human = HumanMessage(
            content=(
                f"需求：{state['user_input']}\n期望：{expected}，实际：{len(task_list)}\n"
                f"任务：{json.dumps([t.model_dump() for t in task_list], ensure_ascii=False)}\n"
                f"结果：{json.dumps({k: v.model_dump() for k, v in task_results.items()}, ensure_ascii=False)}"
            )
        )
        record = _invoke_structured_json(llm, ReflectionRecord, [system, human], metrics)
        record = record.model_copy(update={"round_index": round_index})
    else:
        record = _build_reflection_rule_based(state, round_index)

    logs.append(record)
    can_retry = flags.retry_count < flags.max_retry
    should_retry = not record.passed and can_retry
    update: dict = {"reflection_logs": logs, "run_metrics": metrics}

    if should_retry:
        task_list = list(state.get("task_list") or [])
        task_results = dict(state.get("task_results") or {})
        first_failed_idx = len(task_list)
        for i, task in enumerate(task_list):
            res = _as_task_result(task_results.get(task.task_id), task.task_id)
            if not res.success:
                first_failed_idx = min(first_failed_idx, i)
                task_list[i] = task.model_copy(update={"status": "pending"})
                task_results.pop(task.task_id, None)
        new_flags = _merge_flags(
            state,
            need_retry=True,
            retry_count=flags.retry_count + 1,
            execution_done=False,
            current_task_index=first_failed_idx if first_failed_idx < len(task_list) else 0,
            reflection_done=False,
        )
        msg = (
            f"[反思] 第{round_index}轮未通过 score={record.score}，"
            f"退回执行 retry={new_flags.retry_count}/{new_flags.max_retry}"
        )
        update["task_list"] = task_list
        update["task_results"] = task_results
    else:
        msg = (
            f"[反思] 第{round_index}轮未通过，已达重试上限 {flags.max_retry}"
            if not record.passed and not can_retry
            else f"[反思] 第{round_index}轮通过 score={record.score}"
        )
        new_flags = _merge_flags(state, need_retry=False, reflection_done=True)

    update["execution_flags"] = new_flags
    update["messages"] = [AIMessage(content=msg)]
    return update


def human_review_summary_node(state: AgentState) -> dict:
    if _is_stage_approved(state, "pre_summary"):
        return {"messages": [AIMessage(content="[人工确认] 总结已确认")]}

    if _auto_approve_enabled():
        return _process_human_decision(
            state, "pre_summary", {"approved": True, "comment": "自动批准"}
        )

    logs = state.get("reflection_logs") or []
    decision = interrupt(
        {
            "type": "human_review",
            "stage": "pre_summary",
            "prompt": "确认执行结果后生成报告",
            "task_results": {
                k: v.model_dump() for k, v in (state.get("task_results") or {}).items()
            },
            "reflection": logs[-1].model_dump() if logs else {},
            "thread_id": state.get("thread_id", ""),
        }
    )
    update = _process_human_decision(state, "pre_summary", decision)
    update["execution_flags"] = _merge_flags(
        {**state, **update}, awaiting_human=False, pending_human_stage=None
    )
    return update


def summary_agent_node(state: AgentState) -> dict:
    task_list = state.get("task_list") or []
    task_results = state.get("task_results") or {}
    logs = state.get("reflection_logs") or []
    metrics = compute_run_metrics(state, _get_metrics(state))

    reflection_summary = logs[-1].summary if logs else None
    current_flags = state.get("execution_flags") or ExecutionFlags()
    patch = apply_aggregate_to_state(
        task_list,
        task_results,
        reflection_summary,
        current_flags=current_flags if isinstance(current_flags, ExecutionFlags)
        else ExecutionFlags.model_validate(current_flags),
    )
    new_flags = _merge_flags(state, **patch["execution_flags"].model_dump())

    human_notes = "; ".join(
        f"{a.stage}:{a.comment}" for a in (state.get("human_approvals") or []) if a.comment
    )
    final_output = patch["final_output"]
    final_output += "\n\n" + format_metrics_report(metrics)
    if human_notes:
        final_output += f"\n\n## 人工审批\n{human_notes}"

    return {
        "final_output": final_output,
        "execution_flags": new_flags,
        "run_metrics": metrics,
        "messages": [
            AIMessage(content="[总结] 报告已生成"),
            AIMessage(content=final_output),
        ],
    }


AGENT_NODES = {
    "document_loader": document_loader_agent_node,
    "dispatcher": dispatcher_agent_node,
    "human_review_dispatch": human_review_dispatch_node,
    "executor": executor_agent_node,
    "researcher": researcher_agent_node,
    "reflection": reflection_agent_node,
    "human_review_summary": human_review_summary_node,
    "summary": summary_agent_node,
}
