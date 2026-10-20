from __future__ import annotations

import json
import os
import re
from typing import Any, Literal, Optional, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel

from config import CFG
from data_store import ensure_analytics_db
from external_tools import EXECUTOR_TOOLS, TOOL_MAP
from llm_utils import get_llm, tracked_invoke
from metrics import compute_run_metrics, format_metrics_report, snapshot_pass_rate
from rag_store import format_retrieved_context, ingest_documents, retrieve_context
from skills import match_and_execute_skill
from skills.base import skill_context_from_state
from state import (
    AgentState,
    ExecutionFlags,
    HumanApprovalRecord,
    ReflectionRecord,
    RunMetrics,
    SubTask,
    TaskResult,
)
from supervisor import pick_worker_for_task
from tools import apply_aggregate_to_state, apply_split_to_state, count_expected_tasks

T = TypeVar("T", bound=BaseModel)


def extract_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fenced:
        text = fenced.group(1).strip()
    return json.loads(text)


def call_llm_for_model(
    llm: ChatOpenAI, model_cls: type[T], messages: list[BaseMessage], metrics: RunMetrics
) -> T:
    schema = json.dumps(model_cls.model_json_schema(), ensure_ascii=False)
    system, *rest = messages
    prompt = [
        SystemMessage(
            content=(
                f"{system.content}\n\n"
                "请仅输出一个符合以下 JSON Schema 的 JSON 对象，不要 markdown 或其他说明：\n"
                f"{schema}"
            )
        ),
        *rest,
    ]
    response = tracked_invoke(llm, prompt, node="reflection", metrics=metrics)
    content = response.content if isinstance(response.content, str) else str(response.content)
    return model_cls.model_validate(extract_json_object(content))


def update_execution_flags(state: AgentState, **changes) -> ExecutionFlags:
    current = state.get("execution_flags") or ExecutionFlags()
    if isinstance(current, dict):
        current = ExecutionFlags.model_validate(current)
    merged = current.model_dump()
    merged.update(changes)
    return ExecutionFlags(**merged)


def run_metrics(state: AgentState) -> RunMetrics:
    raw = state.get("run_metrics") or RunMetrics()
    if isinstance(raw, dict):
        return RunMetrics.model_validate(raw)
    return raw


def task_result(raw: TaskResult | dict | None, task_id: str) -> TaskResult:
    if raw is None:
        return TaskResult(task_id=task_id, success=False, error="未执行")
    if isinstance(raw, TaskResult):
        return raw
    if isinstance(raw, dict):
        return TaskResult.model_validate(raw)
    return TaskResult(task_id=task_id, success=False, error="未知结果类型")


def auto_approve() -> bool:
    # 运行时读取，避免 CFG 在 import 时固化导致请求级配置失效
    raw = os.getenv("AUTO_APPROVE_HUMAN")
    return raw is not None and raw.strip().lower() in ("1", "true", "yes", "on")


def append_retrieved_docs(state: AgentState, new_docs: list) -> list:
    from state import RetrievedDoc

    merged = list(state.get("retrieved_context") or [])
    seen = {(doc.source, doc.content[:80]) for doc in merged}
    for raw in new_docs:
        doc = raw if isinstance(raw, RetrievedDoc) else RetrievedDoc.model_validate(raw)
        key = (doc.source, doc.content[:80])
        if key not in seen:
            merged.append(doc)
            seen.add(key)
    return merged


def track_skill(metrics: RunMetrics, skill_id: str) -> None:
    metrics.skill_invocations.append(skill_id)
    metrics.manual_minutes_saved += CFG.skill.manual_minutes.get(skill_id, 10)


def stage_approved(state: AgentState, stage: Literal["post_dispatch", "pre_summary"]) -> bool:
    for record in state.get("human_approvals") or []:
        if record.stage == stage and record.approved:
            return True
    return False


def apply_human_decision(
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
        "execution_flags": update_execution_flags(state, awaiting_human=False, pending_human_stage=None),
        "messages": [AIMessage(content=f"[人工确认/{stage_label}] {status} {comment or ''}")],
    }


def document_loader_agent_node(state: AgentState) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    if flags.documents_loaded:
        return {"messages": [AIMessage(content="[文档加载] 已加载，跳过")]}

    rag_status = ingest_documents()
    db_status = ensure_analytics_db()
    ok = rag_status.get("ok", True) and db_status.get("ok", True)
    message = f"{rag_status.get('message', '')}；{db_status.get('message', '')}"
    if not ok:
        message += "（部分数据源不可用，将降级执行）"
    return {
        "execution_flags": update_execution_flags(state, documents_loaded=True),
        "messages": [AIMessage(content=f"[文档加载] {message}")],
    }


def dispatcher_agent_node(state: AgentState) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    if flags.planning_done and state.get("task_list"):
        return {"messages": [AIMessage(content="[分发] 已拆分，跳过")]}

    split = apply_split_to_state(state["user_input"])
    tasks: list[SubTask] = split["task_list"]
    tasks = [
        task.model_copy(update={"assignee": pick_worker_for_task(task)})  # type: ignore[arg-type]
        for task in tasks
    ]
    return {
        "task_list": tasks,
        "execution_flags": update_execution_flags(
            state,
            planning_done=True,
            current_task_index=0,
        ),
        "messages": [
            AIMessage(
                content=f"[分发] {len(tasks)} 项：" + "、".join(task.title for task in tasks)
            )
        ],
    }


def human_review_dispatch_node(state: AgentState) -> dict:
    if stage_approved(state, "post_dispatch"):
        return {"messages": [AIMessage(content="[人工确认] 分发已确认")]}

    if auto_approve():
        return apply_human_decision(
            state, "post_dispatch", {"approved": True, "comment": "自动批准"}
        )

    tasks = state.get("task_list") or []
    decision = interrupt(
        {
            "type": "human_review",
            "stage": "post_dispatch",
            "prompt": "确认子任务拆分后继续执行",
            "task_list": [task.model_dump() for task in tasks],
            "thread_id": state.get("thread_id", ""),
        }
    )
    update = apply_human_decision(state, "post_dispatch", decision)
    update["execution_flags"] = update_execution_flags(
        {**state, **update}, awaiting_human=False, pending_human_stage=None
    )
    return update


def prefetch_tools_for_task(task: SubTask, metrics: RunMetrics) -> list[str]:
    text = f"{task.title} {task.description}"
    lines: list[str] = []

    if re.search(r"库存|备货|SKU", text):
        result = TOOL_MAP["query_channel_inventory"].invoke({"sku": "SKU-001"})
        lines.append(f"[库存查询] {result}")
        metrics.tool_calls_count += 1

    if re.search(r"搜索|竞品|调研|市场", text):
        result = TOOL_MAP["simple_web_search"].invoke({"query": text[:80]})
        lines.append(f"[市场情报] {result}")
        metrics.tool_calls_count += 1

    if re.search(r"销量|渠道|排行|统计|SQL|数据", text):
        result = TOOL_MAP["analyze_sales_pandas"].invoke(
            {"analysis_type": "channel_compare", "group_by": "channel"}
        )
        lines.append(f"[pandas分析] {result}")
        metrics.tool_calls_count += 1

    return lines


def run_matched_skill(
    task: SubTask, state: AgentState, metrics: RunMetrics, *, prefer: bool = False
) -> TaskResult | None:
    text = f"{task.title} {task.description}"
    ctx = skill_context_from_state(state, task.task_id)
    skill_out = match_and_execute_skill(text, ctx)
    if skill_out is None:
        return None

    track_skill(metrics, skill_out.skill_id)
    failed = "执行失败" in skill_out.summary
    return TaskResult(
        task_id=task.task_id,
        success=not failed,
        output=skill_out.markdown,
        error=skill_out.summary if failed else None,
        tool_calls=[f"skill:{skill_out.skill_id}", *skill_out.data_sources],
    )


def run_llm_with_tools(
    llm: ChatOpenAI,
    messages: list[BaseMessage],
    metrics: RunMetrics,
    node: str,
) -> tuple[str, list[str]]:
    tool_names: list[str] = []
    bound = llm.bind_tools(EXECUTOR_TOOLS)
    conversation = list(messages)

    for _ in range(3):
        response = tracked_invoke(bound, conversation, node=node, metrics=metrics)
        if not getattr(response, "tool_calls", None):
            return str(response.content), tool_names

        conversation.append(response)
        for call in response.tool_calls:
            name = call["name"]
            tool_names.append(name)
            metrics.tool_calls_count += 1
            handler = TOOL_MAP.get(name)
            try:
                result = handler.invoke(call["args"]) if handler else f"未知工具: {name}"
            except Exception as exc:
                result = str(exc)
            conversation.append(ToolMessage(content=str(result), tool_call_id=call["id"]))

    return str(response.content), tool_names


def execute_task(
    task: SubTask,
    state: AgentState,
    llm: Optional[ChatOpenAI],
    *,
    rag_k: int = 4,
    node: str = "executor",
    prefer_skill: bool = False,
) -> TaskResult:
    fail_ids = state.get("simulate_fail_task_ids") or []
    flags = state.get("execution_flags") or ExecutionFlags()
    metrics = run_metrics(state)

    if task.task_id in fail_ids and flags.retry_count < 1:
        return TaskResult(
            task_id=task.task_id,
            success=False,
            error=f"执行失败（{task.task_id}）",
            tool_calls=["simulate_failure"],
        )

    skill_result = run_matched_skill(task, state, metrics, prefer=prefer_skill)
    if skill_result is not None:
        return skill_result

    query = f"{task.title} {task.description} {state['user_input'][:200]}"
    docs = retrieve_context(query, k=rag_k)
    metrics.rag_queries += 1
    policy_context = format_retrieved_context(docs)
    prefetched = prefetch_tools_for_task(task, metrics)

    if llm is not None:
        system = SystemMessage(
            content=(
                "你是快消饮料行业任务执行专家。仅完成当前子任务，输出简洁执行报告。"
                "必须优先依据【业务文档检索结果】与工具返回的数据，不要编造与公司规则冲突的内容。"
                "数据分析任务可调用：query_sales_sql、analyze_sales_pandas、integrate_data_sources。"
                "不要替其他子任务写内容，不要写总方案汇总。"
            )
        )
        human = HumanMessage(
            content=(
                f"总需求背景：{state['user_input']}\n"
                f"当前子任务：{task.model_dump_json(ensure_ascii=False)}\n\n"
                f"【业务文档检索结果】\n{policy_context}\n"
                + ("\n".join(prefetched) if prefetched else "")
            )
        )
        try:
            output, tool_calls = run_llm_with_tools(llm, [system, human], metrics, node=node)
            return TaskResult(
                task_id=task.task_id,
                success=True,
                output=str(output),
                tool_calls=tool_calls or ["llm"],
            )
        except Exception as exc:
            return TaskResult(task_id=task.task_id, success=False, error=str(exc))

    local_output = f"已完成：{task.title}"
    if docs:
        local_output += f"\n（参考文档：{docs[0].source}）"
    return TaskResult(
        task_id=task.task_id,
        success=True,
        output=local_output,
        tool_calls=["local_executor"],
    )


def run_worker_node(
    state: AgentState, *, node: str, rag_k: int, prefer_skill: bool = False
) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    metrics = run_metrics(state)
    tasks: list[SubTask] = list(state.get("task_list") or [])
    results: dict[str, TaskResult] = dict(state.get("task_results") or {})
    idx = flags.current_task_index

    if idx >= len(tasks):
        return {
            "execution_flags": update_execution_flags(state, execution_done=True),
            "run_metrics": metrics,
            "messages": [AIMessage(content=f"[{node}] 全部完成")],
        }

    current = tasks[idx]
    prior = task_result(results.get(current.task_id), current.task_id)
    if flags.retry_count > 0 and prior.success:
        # 重试时只重跑失败项，避免覆盖后续已经成功的结果
        working = list(tasks)
        working[idx] = current.model_copy(update={"status": "success"})
        next_idx = idx + 1
        all_done = next_idx >= len(tasks)
        return {
            "task_list": working,
            "execution_flags": update_execution_flags(
                state,
                current_task_index=next_idx,
                execution_done=all_done,
                need_retry=False,
            ),
            "run_metrics": metrics,
            "messages": [
                AIMessage(
                    content=(
                        f"[{node}] retry={flags.retry_count}/{flags.max_retry} "
                        f"{current.task_id} 跳过（已成功）"
                    )
                )
            ],
        }

    working = list(tasks)
    working[idx] = current.model_copy(update={"status": "running"})

    outcome = execute_task(
        current, state, get_llm(), rag_k=rag_k, node=node, prefer_skill=prefer_skill
    )
    results[current.task_id] = outcome
    final_status = "success" if outcome.success else "failed"
    working[idx] = current.model_copy(update={"status": final_status})

    next_idx = idx + 1
    all_done = next_idx >= len(tasks)
    new_flags = update_execution_flags(
        state,
        current_task_index=next_idx,
        execution_done=all_done,
        need_retry=False,
    )

    if all_done and flags.retry_count == 0:
        snap = {**state, "task_list": working, "task_results": results}
        metrics.first_pass_rate = snapshot_pass_rate(snap)  # type: ignore[arg-type]

    status = "成功" if outcome.success else "失败"
    retry = f" retry={flags.retry_count}/{flags.max_retry}" if flags.retry_count else ""
    detail = (outcome.output or outcome.error or "")[:120]
    retrieved = retrieve_context(f"{current.title} {current.description}", k=2)
    merged_ctx = append_retrieved_docs(
        {**state, "retrieved_context": state.get("retrieved_context") or []},
        retrieved,
    )

    return {
        "task_list": working,
        "task_results": results,
        "execution_flags": new_flags,
        "retrieved_context": merged_ctx,
        "run_metrics": metrics,
        "messages": [AIMessage(content=f"[{node}]{retry} {current.task_id} {status} {detail}")],
    }


def executor_agent_node(state: AgentState) -> dict:
    return run_worker_node(state, node="executor", rag_k=CFG.rag.default_k)


def researcher_agent_node(state: AgentState) -> dict:
    return run_worker_node(state, node="researcher", rag_k=CFG.rag.researcher_k)


def data_analyst_agent_node(state: AgentState) -> dict:
    return run_worker_node(
        state, node="data_analyst", rag_k=CFG.rag.analyst_k, prefer_skill=True
    )


def reflection_from_rules(state: AgentState, round_index: int) -> ReflectionRecord:
    tasks = state.get("task_list") or []
    results = state.get("task_results") or {}
    flags = state.get("execution_flags") or ExecutionFlags()

    failed = [
        task
        for task in tasks
        if not task_result(results.get(task.task_id), task.task_id).success
    ]
    success_count = len(tasks) - len(failed)
    expected = count_expected_tasks(state["user_input"])
    actual = len(tasks)
    coverage_ok = actual >= expected if expected else True

    score = int(success_count / max(len(tasks), 1) * 100)
    if coverage_ok and not failed:
        score = max(score, 95)
    elif not coverage_ok:
        score = min(score, 85)

    issues: list[str] = [f"{task.task_id} 失败" for task in failed]
    if not coverage_ok:
        issues.append(f"子任务拆分不足：期望约 {expected} 项，实际 {actual} 项")

    return ReflectionRecord(
        round_index=round_index,
        passed=len(failed) == 0 and coverage_ok,
        score=score,
        issues=issues,
        suggestions=["重试失败任务"] if failed else [],
        summary=(
            f"{success_count}/{len(tasks)} 成功，期望 {expected} 项，"
            f"重试 {flags.retry_count}/{flags.max_retry}"
        ),
    )


def reflection_agent_node(state: AgentState) -> dict:
    flags = state.get("execution_flags") or ExecutionFlags()
    metrics = run_metrics(state)
    logs = list(state.get("reflection_logs") or [])
    round_index = len(logs) + 1

    llm = get_llm()
    if llm:
        tasks = state.get("task_list") or []
        results = state.get("task_results") or {}
        expected = count_expected_tasks(state["user_input"])
        system = SystemMessage(
            content=(
                "质量审查员。存在失败任务时 passed 应为 false。"
                "若所有子任务 success 且实际子任务数 >= 期望条目数，passed 应为 true，score >= 95。"
            )
        )
        human = HumanMessage(
            content=(
                f"需求：{state['user_input']}\n期望：{expected}，实际：{len(tasks)}\n"
                f"任务：{json.dumps([task.model_dump() for task in tasks], ensure_ascii=False)}\n"
                f"结果：{json.dumps({k: v.model_dump() for k, v in results.items()}, ensure_ascii=False)}"
            )
        )
        llm_record = call_llm_for_model(llm, ReflectionRecord, [system, human], metrics)
        rule_record = reflection_from_rules(state, round_index)
        if not rule_record.passed:
            # 规则发现失败项时，不允许 LLM 把结果判为通过
            llm_record = llm_record.model_copy(
                update={
                    "passed": False,
                    "score": min(llm_record.score, rule_record.score),
                    "issues": list(
                        dict.fromkeys([*llm_record.issues, *rule_record.issues])
                    ),
                    "suggestions": list(
                        dict.fromkeys(
                            [*llm_record.suggestions, *rule_record.suggestions]
                        )
                    ),
                }
            )
        record = llm_record.model_copy(update={"round_index": round_index})
    else:
        record = reflection_from_rules(state, round_index)

    logs.append(record)
    can_retry = flags.retry_count < flags.max_retry
    should_retry = not record.passed and can_retry
    update: dict = {"reflection_logs": logs, "run_metrics": metrics}

    if should_retry:
        tasks = list(state.get("task_list") or [])
        results = dict(state.get("task_results") or {})
        first_failed_idx = len(tasks)
        for i, task in enumerate(tasks):
            outcome = task_result(results.get(task.task_id), task.task_id)
            if not outcome.success:
                first_failed_idx = min(first_failed_idx, i)
                tasks[i] = task.model_copy(update={"status": "pending"})
                results.pop(task.task_id, None)
        new_flags = update_execution_flags(
            state,
            need_retry=True,
            retry_count=flags.retry_count + 1,
            execution_done=False,
            current_task_index=first_failed_idx if first_failed_idx < len(tasks) else 0,
            reflection_done=False,
        )
        msg = (
            f"[反思] 第{round_index}轮未通过 score={record.score}，"
            f"退回执行 retry={new_flags.retry_count}/{new_flags.max_retry}"
        )
        update["task_list"] = tasks
        update["task_results"] = results
    else:
        msg = (
            f"[反思] 第{round_index}轮未通过，已达重试上限 {flags.max_retry}"
            if not record.passed and not can_retry
            else f"[反思] 第{round_index}轮通过 score={record.score}"
        )
        new_flags = update_execution_flags(state, need_retry=False, reflection_done=True)

    update["execution_flags"] = new_flags
    update["messages"] = [AIMessage(content=msg)]
    return update


def human_review_summary_node(state: AgentState) -> dict:
    if stage_approved(state, "pre_summary"):
        return {"messages": [AIMessage(content="[人工确认] 总结已确认")]}

    if auto_approve():
        return apply_human_decision(
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
    update = apply_human_decision(state, "pre_summary", decision)
    update["execution_flags"] = update_execution_flags(
        {**state, **update}, awaiting_human=False, pending_human_stage=None
    )
    return update


def summary_agent_node(state: AgentState) -> dict:
    tasks = state.get("task_list") or []
    results = state.get("task_results") or {}
    logs = state.get("reflection_logs") or []
    metrics = compute_run_metrics(state, run_metrics(state))

    reflection_summary = logs[-1].summary if logs else None
    current_flags = state.get("execution_flags") or ExecutionFlags()
    patch = apply_aggregate_to_state(
        tasks,
        results,
        reflection_summary,
        current_flags=current_flags
        if isinstance(current_flags, ExecutionFlags)
        else ExecutionFlags.model_validate(current_flags),
    )
    new_flags = update_execution_flags(state, **patch["execution_flags"].model_dump())

    human_notes = "; ".join(
        f"{approval.stage}:{approval.comment}"
        for approval in (state.get("human_approvals") or [])
        if approval.comment
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
            AIMessage(content="[总结] 周报已生成"),
            AIMessage(content=final_output),
        ],
    }
