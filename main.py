"""快消数据分析 Agent：命令行入口与集成测试。"""
from __future__ import annotations

import os
import sqlite3
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Optional

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

from dotenv import load_dotenv
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from llm_utils import get_llm_status
from graph import build_graph, get_app
from paths import save_user_report
from state import AgentState, ExecutionFlags, SubTask, TaskResult, create_checkpoint_serializer, create_initial_state

CHECKPOINT_DIR = Path(__file__).parent / "data"
CHECKPOINT_DB = CHECKPOINT_DIR / "checkpoints.db"


@contextmanager
def without_llm() -> Iterator[None]:
    """内置测试走规则逻辑，避免 LLM 输出不稳定导致断言失败"""
    saved = {k: os.environ.pop(k, None) for k in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY")}
    os.environ["USE_SUPERVISOR"] = "false"
    try:
        yield
    finally:
        for k, v in saved.items():
            if v is not None:
                os.environ[k] = v
        os.environ.pop("USE_SUPERVISOR", None)


@contextmanager
def get_checkpointer(db_path: Path = CHECKPOINT_DB) -> Iterator[SqliteSaver]:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    serde = create_checkpoint_serializer()
    try:
        yield SqliteSaver(conn, serde=serde)
    finally:
        conn.close()


def make_run_config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def run_until_pause_or_finish(
    user_input: str,
    *,
    thread_id: Optional[str] = None,
    max_retry: int = 2,
    simulate_fail_task_ids: Optional[list[str]] = None,
    checkpointer: Optional[SqliteSaver] = None,
    resume_payload: Optional[Any] = None,
    verbose: bool = True,
) -> tuple[AgentState, bool, Optional[dict]]:
    app = get_app(checkpointer=checkpointer)
    tid = thread_id or str(os.urandom(4).hex())
    config = make_run_config(tid)

    if resume_payload is not None:
        inputs: Any = Command(resume=resume_payload)
    else:
        inputs = create_initial_state(
            user_input,
            thread_id=tid,
            max_retry=max_retry,
            simulate_fail_task_ids=simulate_fail_task_ids,
        )

    if verbose and resume_payload is None:
        print(f"thread_id={tid}")
        print(f"input: {user_input}")

    interrupt_payload: Optional[dict] = None

    for step in app.stream(inputs, config, stream_mode="updates"):
        if verbose:
            for node_name, node_output in step.items():
                for msg in node_output.get("messages", []):
                    content = getattr(msg, "content", str(msg))
                    print(f"  [{node_name}] {content}")

    snapshot = app.get_state(config)

    if snapshot.next:
        interrupts = getattr(snapshot, "interrupts", ()) or ()
        if interrupts:
            interrupt_payload = interrupts[0].value if hasattr(interrupts[0], "value") else interrupts[0]
        finished = False
    else:
        finished = True

    final_values = snapshot.values if snapshot.values else inputs
    if isinstance(final_values, dict) and "user_input" in final_values:
        state: AgentState = final_values  # type: ignore[assignment]
    else:
        state = app.invoke(None, config) if finished else create_initial_state(user_input, thread_id=tid)

    if finished and verbose:
        _print_final_report(state)

    return state, finished, interrupt_payload


def resume_run(
    thread_id: str,
    approval: dict,
    *,
    checkpointer: Optional[SqliteSaver] = None,
    verbose: bool = True,
) -> tuple[AgentState, bool, Optional[dict]]:
    return run_until_pause_or_finish(
        user_input="",
        thread_id=thread_id,
        checkpointer=checkpointer,
        resume_payload=approval,
        verbose=verbose,
    )


def run(user_input: str, *, verbose: bool = True, max_retry: int = 2) -> AgentState:
    os.environ.setdefault("AUTO_APPROVE_HUMAN", "true")
    with get_checkpointer() as cp:
        state, finished, _ = run_until_pause_or_finish(
            user_input, checkpointer=cp, max_retry=max_retry, verbose=verbose
        )
        tid = state.get("thread_id", "")
        while not finished:
            state, finished, _ = resume_run(
                tid, {"approved": True, "comment": "续跑"}, checkpointer=cp, verbose=verbose
            )
        output = state.get("final_output")
        if verbose and output:
            saved = save_user_report(output, tid)
            print(f"\n报告已保存: {saved}")
        return state


def _print_final_report(state: AgentState) -> None:
    flags = state.get("execution_flags") or ExecutionFlags()
    print("\n--- 结果 ---")
    print(f"thread_id={state.get('thread_id', '')}")
    print(f"retry={flags.retry_count}/{flags.max_retry} finished={flags.is_finished}")
    for task in state.get("task_list") or []:
        print(f"  {task.task_id} {task.title} [{task.status}]")
    print(state.get("final_output") or "")


def _use_test_checkpoint_db(test_name: str) -> Path:
    path = CHECKPOINT_DIR / f"test_{test_name}.db"
    if path.exists():
        path.unlink()
    return path


def test_split_numbered_tasks() -> None:
    from tools import _parse_user_input_to_tasks, count_expected_tasks

    demo = (
        "元气森林华东区渠道分析，需要完成："
        "1. 统计便利店、商超、电商、特通渠道销量与销售额对比；"
        "2. 输出气泡水及电解质水 SKU 销售 TOP 排行；"
        "3. 检索渠道陈列费与定价政策；"
        "4. 整合 SQL 销售数据、pandas 统计与政策文档生成周报；"
        "5. 查询核心 SKU 渠道库存；"
        "6. 调研竞品无糖气泡水市场动态。"
    )
    assert count_expected_tasks(demo) == 6
    tasks = _parse_user_input_to_tasks(demo)
    assert len(tasks) == 6
    assert all(t.title for t in tasks)
    assert "竞品" in tasks[-1].title
    print("PASS test_split_numbered_tasks")


def test_graph_node_names() -> None:
    os.environ["USE_SUPERVISOR"] = "false"
    app = build_graph()
    expected = {
        "document_loader", "dispatcher", "human_review_dispatch", "executor",
        "reflection", "human_review_summary", "summary",
    }
    actual = set(app.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert expected.issubset(actual)
    print("PASS test_graph_node_names")


def test_supervisor_graph_nodes() -> None:
    from graph import build_supervisor_graph

    app = build_supervisor_graph()
    nodes = set(app.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert "supervisor" in nodes
    assert "researcher" in nodes
    assert "data_analyst" in nodes
    print("PASS test_supervisor_graph_nodes")


def test_rag_ingest_and_retrieve() -> None:
    from rag_store import ingest_documents, retrieve_context

    result = ingest_documents(force=True)
    assert result["loaded"] == 3
    assert "genki_analysis_playbook.md" in result.get("files", [])
    docs = retrieve_context("渠道陈列费用政策", k=2)
    assert len(docs) >= 1
    assert len(docs[0].content) > 10
    print("PASS test_rag_ingest_and_retrieve")


def test_external_tools() -> None:
    from external_tools import query_channel_inventory, read_local_file, simple_web_search

    inv = query_channel_inventory.invoke({"sku": "YQ-001"})
    assert "stock" in inv
    search = simple_web_search.invoke({"query": "竞品气泡水"})
    assert "竞品" in search or "气泡水" in search
    doc = read_local_file.invoke({"path": "fmcg_channel_policy.md"})
    assert "渠道" in doc
    print("PASS test_external_tools")


def test_data_analytics() -> None:
    from data_store import ensure_analytics_db, run_readonly_sql
    from data_tools import analyze_sales_pandas, query_sales_sql

    init = ensure_analytics_db()
    assert init["rows"] >= 20

    sql_out = query_sales_sql.invoke(
        {"sql": "SELECT channel, SUM(revenue) AS revenue FROM channel_sales GROUP BY channel"}
    )
    assert "preview" in sql_out

    pandas_out = analyze_sales_pandas.invoke({"analysis_type": "ranking", "group_by": "channel"})
    assert "summary" in pandas_out
    df = run_readonly_sql("SELECT COUNT(*) AS cnt FROM channel_sales")
    assert int(df.iloc[0]["cnt"]) >= 20
    print("PASS test_data_analytics")


def test_skill_registry() -> None:
    from skills import list_skills, match_and_execute_skill

    skills = list_skills()
    assert len(skills) >= 5
    out = match_and_execute_skill("统计各渠道销量对比")
    assert out is not None
    assert out.skill_id == "channel_compare"
    assert "数据概览" in out.markdown
    print("PASS test_skill_registry")


def test_supervisor_routing() -> None:
    from supervisor import decide_supervisor_route, pick_worker_for_task
    from state import create_initial_state, ExecutionFlags, SubTask

    state = create_initial_state("测试")
    assert decide_supervisor_route(state) == "document_loader"

    state["execution_flags"] = ExecutionFlags(documents_loaded=True)
    assert decide_supervisor_route(state) == "dispatcher"

    task = SubTask(task_id="T001", title="检索渠道陈列费用政策", description="")
    assert pick_worker_for_task(task) == "researcher"
    task2 = SubTask(task_id="T002", title="调研竞品气泡水市场动态", description="")
    assert pick_worker_for_task(task2) == "executor"
    task3 = SubTask(task_id="T003", title="统计各渠道销量排行", description="")
    assert pick_worker_for_task(task3) == "data_analyst"
    print("PASS test_supervisor_routing")


def test_reflection_retry_with_simulated_failure() -> None:
    os.environ["AUTO_APPROVE_HUMAN"] = "true"
    db = _use_test_checkpoint_db("retry")
    tid = "retry-demo-thread"

    with get_checkpointer(db) as cp:
        state, finished, _ = run_until_pause_or_finish(
            "任务A、任务B、任务C",
            thread_id=tid,
            simulate_fail_task_ids=["T002"],
            max_retry=2,
            checkpointer=cp,
            verbose=False,
        )
        while not finished:
            state, finished, _ = resume_run(
                tid, {"approved": True}, checkpointer=cp, verbose=False
            )

    flags = state["execution_flags"]
    logs = state["reflection_logs"]

    assert len(logs) >= 2
    assert not logs[0].passed
    assert logs[-1].passed
    assert flags.retry_count >= 1
    assert flags.retry_count <= flags.max_retry

    t002 = state["task_results"]["T002"]
    if isinstance(t002, dict):
        t002 = TaskResult.model_validate(t002)
    assert t002.success
    assert flags.is_finished
    print(f"PASS test_reflection_retry retry_count={flags.retry_count}")


def test_sqlite_checkpoint_resume() -> None:
    os.environ["AUTO_APPROVE_HUMAN"] = "false"
    db = _use_test_checkpoint_db("checkpoint")
    user_input = "编写测试用例、执行回归测试、输出测试报告"
    tid = "checkpoint-demo-thread"

    with get_checkpointer(db) as cp:
        app = build_graph(checkpointer=cp)
        config = make_run_config(tid)
        initial = create_initial_state(user_input, thread_id=tid, max_retry=2)
        for _ in app.stream(initial, config, stream_mode="updates"):
            pass
        assert app.get_state(config).next

    with get_checkpointer(db) as cp:
        state, finished, _ = run_until_pause_or_finish(
            user_input,
            thread_id=tid,
            checkpointer=cp,
            resume_payload={"approved": True, "comment": "确认分发"},
            verbose=False,
        )
        while not finished:
            state, finished, _ = resume_run(
                tid, {"approved": True, "comment": "确认总结"}, checkpointer=cp, verbose=False
            )

    assert len(state.get("human_approvals") or []) >= 2
    assert state["execution_flags"].is_finished
    assert "任务调度汇总报告" in state["final_output"] or "业务分析周报" in state["final_output"]
    print("PASS test_sqlite_checkpoint_resume")


def test_human_in_the_loop_interrupt() -> None:
    os.environ["AUTO_APPROVE_HUMAN"] = "false"
    db = _use_test_checkpoint_db("hitl")
    tid = "hitl-demo-thread"

    with get_checkpointer(db) as cp:
        app = build_graph(checkpointer=cp)
        config = make_run_config(tid)
        initial = create_initial_state("统计渠道销量、输出SKU排行、检索陈列政策", thread_id=tid)

        for _ in app.stream(initial, config, stream_mode="updates"):
            pass
        assert app.get_state(config).next

        for _ in app.stream(
            Command(resume={"approved": True, "comment": "确认分发"}),
            config,
            stream_mode="updates",
        ):
            pass

        snap = app.get_state(config)
        approvals = (snap.values or {}).get("human_approvals") or []
        assert any(a.stage == "post_dispatch" and a.approved for a in approvals)

        while snap.next:
            for _ in app.stream(
                Command(resume={"approved": True, "comment": "确认总结"}),
                config,
                stream_mode="updates",
            ):
                pass
            snap = app.get_state(config)

        final = snap.values
        assert final["execution_flags"].is_finished
        assert len(final.get("human_approvals") or []) >= 2

    print("PASS test_human_in_the_loop_interrupt")


def test_max_retry_exhausted() -> None:
    os.environ["AUTO_APPROVE_HUMAN"] = "true"
    from agents import reflection_agent_node

    state = create_initial_state("测试", max_retry=1)
    state["task_list"] = [SubTask(task_id="T001", title="失败任务", status="failed")]
    state["task_results"] = {"T001": TaskResult(task_id="T001", success=False, error="失败")}
    state["execution_flags"] = ExecutionFlags(
        planning_done=True, execution_done=True, max_retry=1, retry_count=1
    )

    update = reflection_agent_node(state)
    flags = update["execution_flags"]
    assert not flags.need_retry
    assert flags.reflection_done
    print("PASS test_max_retry_exhausted")


def test_simple_flow() -> None:
    os.environ["AUTO_APPROVE_HUMAN"] = "true"
    final = run("分析竞品、制定方案、输出排期", verbose=False)
    assert final["execution_flags"].is_finished
    assert len(final.get("human_approvals") or []) >= 2
    print("PASS test_simple_flow")


if __name__ == "__main__":
    load_dotenv()

    print(f"LLM: {get_llm_status()}")
    print(f"checkpoint: {CHECKPOINT_DB}\n")

    with without_llm():
        test_split_numbered_tasks()
        test_graph_node_names()
        test_supervisor_graph_nodes()
        test_rag_ingest_and_retrieve()
        test_data_analytics()
        test_skill_registry()
        test_external_tools()
        test_supervisor_routing()
        test_reflection_retry_with_simulated_failure()
        test_max_retry_exhausted()
        test_human_in_the_loop_interrupt()
        test_sqlite_checkpoint_resume()
        test_simple_flow()

    print("\n--- demo ---")
    os.environ["AUTO_APPROVE_HUMAN"] = "true"
    run(
        "元气森林华东区渠道分析，需要完成："
        "1. 统计便利店、商超、电商、特通渠道销量与销售额对比；"
        "2. 输出气泡水及电解质水 SKU 销售 TOP 排行；"
        "3. 检索渠道陈列费与定价政策；"
        "4. 整合 SQL 销售数据、pandas 统计与政策文档生成周报；"
        "5. 查询核心 SKU 渠道库存；"
        "6. 调研竞品无糖气泡水市场动态。",
        verbose=True,
    )
