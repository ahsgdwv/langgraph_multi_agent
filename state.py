"""全局状态：Pydantic 模型 + LangGraph AgentState。"""
from __future__ import annotations

import uuid
from typing import Annotated, Literal, Optional, TYPE_CHECKING

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

if TYPE_CHECKING:
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


class SubTask(BaseModel):
    task_id: str = Field(..., description="子任务 ID")
    title: str = Field(..., description="标题")
    description: str = Field(default="", description="描述")
    priority: Literal["low", "medium", "high"] = Field(default="medium")
    assignee: Literal["executor", "researcher", "data_analyst"] = Field(default="executor")
    status: Literal["pending", "running", "success", "failed"] = Field(default="pending")


class TaskResult(BaseModel):
    """子任务执行结果"""

    task_id: str
    success: bool
    output: str = ""
    error: Optional[str] = None
    tool_calls: list[str] = Field(default_factory=list)


class ReflectionRecord(BaseModel):
    """反思校验记录"""

    round_index: int = Field(..., ge=1)
    passed: bool
    score: int = Field(..., ge=0, le=100)
    issues: list[str] = Field(default_factory=list)
    suggestions: list[str] = Field(default_factory=list)
    summary: str = ""


class HumanApprovalRecord(BaseModel):
    """人工审批记录"""

    stage: Literal["post_dispatch", "pre_summary"]
    approved: bool
    comment: str = ""
    reviewer: str = "human"


class ExecutionFlags(BaseModel):
    """流程执行标记"""

    planning_done: bool = False
    execution_done: bool = False
    reflection_done: bool = False
    need_retry: bool = False
    current_task_index: int = 0
    max_retry: int = Field(default=2, ge=0)
    retry_count: int = Field(default=0, ge=0)
    is_finished: bool = False
    awaiting_human: bool = False
    pending_human_stage: Optional[Literal["post_dispatch", "pre_summary"]] = None
    documents_loaded: bool = False


class RetrievedDoc(BaseModel):
    """RAG 检索片段"""

    source: str
    content: str
    score: float = 0.0


class LLMCallLog(BaseModel):
    """单次 LLM 调用日志"""

    node: str
    model: str
    latency_ms: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class RunMetrics(BaseModel):
    first_pass_rate: float = 0.0
    final_pass_rate: float = 0.0
    improvement_pct: float = 0.0
    llm_calls: list[LLMCallLog] = Field(default_factory=list)
    tool_calls_count: int = 0
    rag_queries: int = 0
    skill_invocations: list[str] = Field(default_factory=list)
    manual_minutes_saved: int = 0


MSGPACK_SERIALIZABLE_MODELS: tuple[type[BaseModel], ...] = (
    SubTask,
    TaskResult,
    ReflectionRecord,
    HumanApprovalRecord,
    ExecutionFlags,
    RetrievedDoc,
    LLMCallLog,
    RunMetrics,
)


def create_checkpoint_serializer() -> "JsonPlusSerializer":
    """创建 Checkpoint 序列化器，注册自定义 Pydantic 类型。"""
    from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

    return JsonPlusSerializer(allowed_msgpack_modules=MSGPACK_SERIALIZABLE_MODELS)


class AgentState(TypedDict):
    """LangGraph 全局状态"""

    user_input: str
    task_list: list[SubTask]
    task_results: dict[str, TaskResult]
    reflection_logs: list[ReflectionRecord]
    human_approvals: list[HumanApprovalRecord]
    final_output: str
    execution_flags: ExecutionFlags
    thread_id: str
    simulate_fail_task_ids: list[str]
    messages: Annotated[list[BaseMessage], add_messages]
    retrieved_context: list[RetrievedDoc]
    run_metrics: RunMetrics
    supervisor_decision: str
    use_supervisor: bool


def create_initial_state(
    user_input: str,
    *,
    thread_id: Optional[str] = None,
    max_retry: int = 2,
    simulate_fail_task_ids: Optional[list[str]] = None,
    use_supervisor: bool = True,
) -> AgentState:
    """构造初始 State"""
    return AgentState(
        user_input=user_input,
        task_list=[],
        task_results={},
        reflection_logs=[],
        human_approvals=[],
        final_output="",
        execution_flags=ExecutionFlags(max_retry=max_retry),
        thread_id=thread_id or str(uuid.uuid4()),
        simulate_fail_task_ids=simulate_fail_task_ids or [],
        messages=[],
        retrieved_context=[],
        run_metrics=RunMetrics(),
        supervisor_decision="",
        use_supervisor=use_supervisor,
    )


if __name__ == "__main__":
    s = create_initial_state("统计渠道销量、输出SKU排行、检索陈列政策")
    print(f"thread_id={s['thread_id']}, max_retry={s['execution_flags'].max_retry}")
