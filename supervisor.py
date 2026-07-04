"""Supervisor 关键词路由。"""
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
    "data_analyst",
    "reflection",
    "human_review_summary",
    "summary",
    "__end__",
]

_RESEARCHER_KEYWORDS = re.compile(
    r"政策|文档|陈列|定价|合规|费用|进场|规则|FAQ|赔付"
)
_DATA_ANALYST_KEYWORDS = re.compile(
    r"销量|销售额|同比|环比|SKU|排行|数据分析|统计|报表|SQL|pandas|整合|周报|月报|复盘|TOP|对比.*渠道|渠道.*对比|渠道.*销量"
)


def pick_worker_for_task(task: SubTask) -> Literal["executor", "researcher", "data_analyst"]:
    """按任务关键词选择执行 Agent（政策类优先于数据统计类）。"""
    text = f"{task.title} {task.description}"
    if _RESEARCHER_KEYWORDS.search(text):
        return "researcher"
    if _DATA_ANALYST_KEYWORDS.search(text):
        return "data_analyst"
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
    if route in ("executor", "researcher", "data_analyst") and flags.current_task_index < len(task_list):
        task = task_list[flags.current_task_index]
        detail = f" -> {task.task_id}:{task.title[:30]}"
    return {
        "supervisor_decision": route,
        "messages": [AIMessage(content=f"[Supervisor] 路由 {route}{detail}")],
    }
