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

# 库存/备货属于执行类，优先于 SKU 等数据关键词
_INVENTORY_KW = re.compile(r"库存|备货|补货|在库|可用库存|仓储|WMS", re.I)
# 政策类优先于数据统计类
_RESEARCHER_KW = re.compile(r"政策|文档|陈列|定价|合规|费用|进场|规则|FAQ|赔付")
_DATA_ANALYST_KW = re.compile(
    r"销量|销售额|同比|环比|SKU|排行|数据分析|统计|报表|SQL|pandas|整合|周报|月报|复盘|TOP|对比.*渠道|渠道.*对比|渠道.*销量"
)


def pick_worker_for_task(task: SubTask) -> Literal["executor", "researcher", "data_analyst"]:
    text = f"{task.title} {task.description}"
    if _INVENTORY_KW.search(text):
        return "executor"
    if _RESEARCHER_KW.search(text):
        return "researcher"
    if _DATA_ANALYST_KW.search(text):
        return "data_analyst"
    return "executor"


def decide_supervisor_route(state: AgentState) -> SupervisorRoute:
    run_flags = state.get("execution_flags") or ExecutionFlags()
    子任务 = state.get("task_list") or []

    if not run_flags.documents_loaded:
        return "document_loader"
    if not run_flags.planning_done:
        return "dispatcher"
    if run_flags.planning_done and not _dispatch_approved(state):
        return "human_review_dispatch"
    if not run_flags.execution_done:
        idx = run_flags.current_task_index
        if idx < 0 or idx >= len(子任务):
            return "reflection"
        return pick_worker_for_task(子任务[idx])
    if not run_flags.reflection_done:
        return "reflection"
    if run_flags.reflection_done and not _summary_approved(state):
        return "human_review_summary"
    if run_flags.is_finished:
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
    route = decide_supervisor_route(state)
    子任务 = state.get("task_list") or []
    run_flags = state.get("execution_flags") or ExecutionFlags()

    detail = ""
    if route in ("executor", "researcher", "data_analyst"):
        idx = run_flags.current_task_index
        if 0 <= idx < len(子任务):
            task = 子任务[idx]
            detail = f" -> {task.task_id}:{task.title[:30]}"

    return {
        "supervisor_decision": route,
        "messages": [AIMessage(content=f"[Supervisor] 路由 {route}{detail}")],
    }
