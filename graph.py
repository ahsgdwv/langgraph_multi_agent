"""LangGraph 图：经典流程 + Supervisor 版。"""
from __future__ import annotations

import os
from typing import Literal, Optional

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph

from agents import (
    data_analyst_agent_node,
    dispatcher_agent_node,
    document_loader_agent_node,
    executor_agent_node,
    human_review_dispatch_node,
    human_review_summary_node,
    reflection_agent_node,
    researcher_agent_node,
    summary_agent_node,
)
from state import AgentState, ExecutionFlags
from supervisor import decide_supervisor_route, supervisor_agent_node


def route_after_executor(state: AgentState) -> Literal["executor", "reflection"]:
    flags = state.get("execution_flags") or ExecutionFlags()
    return "reflection" if flags.execution_done else "executor"


def route_after_reflection(state: AgentState) -> Literal["executor", "human_review_summary"]:
    flags = state.get("execution_flags") or ExecutionFlags()
    if flags.need_retry and not flags.execution_done:
        return "executor"
    return "human_review_summary"


def route_supervisor(state: AgentState) -> str:
    return state.get("supervisor_decision") or decide_supervisor_route(state)


def build_graph(checkpointer: Optional[SqliteSaver] = None):
    """经典固定流程（含文档加载）。"""
    return _build_classic_simple(checkpointer)


def _build_classic_simple(checkpointer: Optional[SqliteSaver] = None):
    workflow = StateGraph(AgentState)
    workflow.add_node("document_loader", document_loader_agent_node)
    workflow.add_node("dispatcher", dispatcher_agent_node)
    workflow.add_node("human_review_dispatch", human_review_dispatch_node)
    workflow.add_node("executor", executor_agent_node)
    workflow.add_node("reflection", reflection_agent_node)
    workflow.add_node("human_review_summary", human_review_summary_node)
    workflow.add_node("summary", summary_agent_node)

    workflow.set_entry_point("document_loader")
    workflow.add_edge("document_loader", "dispatcher")
    workflow.add_edge("dispatcher", "human_review_dispatch")
    workflow.add_edge("human_review_dispatch", "executor")
    workflow.add_conditional_edges(
        "executor",
        route_after_executor,
        {"executor": "executor", "reflection": "reflection"},
    )
    workflow.add_conditional_edges(
        "reflection",
        route_after_reflection,
        {"executor": "executor", "human_review_summary": "human_review_summary"},
    )
    workflow.add_edge("human_review_summary", "summary")
    workflow.add_edge("summary", END)
    return workflow.compile(checkpointer=checkpointer)


def build_supervisor_graph(checkpointer: Optional[SqliteSaver] = None):
    """Supervisor 动态多 Agent 调度图。"""
    workflow = StateGraph(AgentState)

    workflow.add_node("supervisor", supervisor_agent_node)
    workflow.add_node("document_loader", document_loader_agent_node)
    workflow.add_node("dispatcher", dispatcher_agent_node)
    workflow.add_node("human_review_dispatch", human_review_dispatch_node)
    workflow.add_node("executor", executor_agent_node)
    workflow.add_node("researcher", researcher_agent_node)
    workflow.add_node("data_analyst", data_analyst_agent_node)
    workflow.add_node("reflection", reflection_agent_node)
    workflow.add_node("human_review_summary", human_review_summary_node)
    workflow.add_node("summary", summary_agent_node)

    workflow.set_entry_point("supervisor")
    workflow.add_conditional_edges(
        "supervisor",
        route_supervisor,
        {
            "document_loader": "document_loader",
            "dispatcher": "dispatcher",
            "human_review_dispatch": "human_review_dispatch",
            "executor": "executor",
            "researcher": "researcher",
            "data_analyst": "data_analyst",
            "reflection": "reflection",
            "human_review_summary": "human_review_summary",
            "summary": "summary",
        },
    )

    for node in (
        "document_loader",
        "dispatcher",
        "human_review_dispatch",
        "executor",
        "researcher",
        "data_analyst",
        "reflection",
        "human_review_summary",
    ):
        workflow.add_edge(node, "supervisor")

    workflow.add_edge("summary", END)
    return workflow.compile(checkpointer=checkpointer)


def get_app(checkpointer: Optional[SqliteSaver] = None):
    use_supervisor = os.getenv("USE_SUPERVISOR", "true").lower() in ("1", "true", "yes")
    if use_supervisor:
        return build_supervisor_graph(checkpointer)
    return _build_classic_simple(checkpointer)
