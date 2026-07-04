"""Supervisor 调度 Agent：根据任务状态动态选择下一节点。"""
from __future__ import annotations

import re
from typing import Literal

from langchain_core.messages import AIMessage

from state import AgentState, ExecutionFlags, SubTask

SupervisorRoute = Literal[
    "document_loader",
    "dispatcher",
    "human_review_dispatch",
    "executor",
    "researcher",
    "reflection",
    "human_review_summary",
    "summary",
    "__end__",
]

_RESEARCHER_KEYWORDS = re.compile(
    r"规则|文档|政策|问答|赔付|满减|优惠券|库存|客服|售后|合规"
)
_TOOL_KEYWORDS = re.compile(r"库存|竞品|搜索|调研|查询|预估|核算")


def pick_worker_for_task(task: SubTask) -> Literal["executor", "researcher"]:
    """按任务复杂度与关键词选择执行 Agent。"""
    text = f"{task.title} {task.description}"
    if _RESEARCHER_KEYWORDS.search(text):
        return "researcher"
    return "executor"


def decide_supervisor_route(state: AgentState) -> SupervisorRoute:
    """Supervisor 路由决策。"""
    flags = state.get("execution_flags") or ExecutionFlags()
    task_list = state.get("task_list") or []

    if not flags.documents_loaded:
        return "document_loader"
    if not flags.planning_done:
        return "dispatcher"
    if flags.planning_done and not _dispatch_approved(state):
        return "human_review_dispatch"
    if not flags.execution_done:
        return pick_worker_for_task(task_list[flags.current_task_index])
    if not flags.reflection_done:
        return "reflection"
    if flags.reflection_done and not _summary_approved(state):
        return "human_review_summary"
    if flags.is_finished:
        return "__end__"
    return "summary"


def _dispatch_approved(state: AgentState) -> bool:
    for record in state.get("human_approvals") or []:
        if record.stage == "post_dispatch" and record.approved:
            return True
    return False


def _summary_approved(state: AgentState) -> bool:
    for record in state.get("human_approvals") or []:
        if record.stage == "pre_summary" and record.approved:
            return True
    return False


def supervisor_agent_node(state: AgentState) -> dict:
    """Supervisor 节点：写入下一跳决策。"""
    route = decide_supervisor_route(state)
    task_list = state.get("task_list") or []
    flags = state.get("execution_flags") or ExecutionFlags()
    detail = ""
    if route in ("executor", "researcher") and flags.current_task_index < len(task_list):
        task = task_list[flags.current_task_index]
        detail = f" -> {task.task_id}:{task.title[:30]}"
    return {
        "supervisor_decision": route,
        "messages": [AIMessage(content=f"[Supervisor] 路由 {route}{detail}")],
    }
