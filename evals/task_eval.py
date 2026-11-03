from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path
from uuid import uuid4

from main import get_checkpointer, resume_run, run_until_pause_or_finish
from skills import match_and_execute_skill
from state import SubTask
from supervisor import pick_worker_for_task


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def _text_score(text: str, terms: list[str]) -> bool:
    haystack = (text or "").lower()
    return all(str(term).lower() in haystack for term in terms)


def _source_score(sources: list[str], expected: list[str]) -> bool:
    return all(any(source in actual for actual in sources) for source in expected)


def evaluate_skill_case(case: dict) -> dict:
    started = time.perf_counter()
    task = SubTask(
        task_id=case["id"],
        title=case["input"],
        description=case["input"],
    )
    actual_worker = pick_worker_for_task(task)
    output = match_and_execute_skill(case["input"])
    actual_skill = output.skill_id if output else None
    text = output.markdown if output else ""
    sources = output.data_sources if output else []
    expected_skill = case.get("expected_skill")
    expected_terms = case.get("expected_terms", [])
    expected_sources = case.get("expected_sources", [])

    routing_ok = actual_worker == case.get("expected_worker")
    skill_ok = actual_skill == expected_skill
    terms_ok = _text_score(text, expected_terms)
    sources_ok = _source_score(sources, expected_sources)
    success = routing_ok and skill_ok and terms_ok and sources_ok
    return {
        "id": case["id"],
        "type": "skill",
        "success": success,
        "routing_ok": routing_ok,
        "skill_ok": skill_ok,
        "terms_ok": terms_ok,
        "sources_ok": sources_ok,
        "expected_worker": case.get("expected_worker"),
        "actual_worker": actual_worker,
        "expected_skill": expected_skill,
        "actual_skill": actual_skill,
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def evaluate_workflow_case(case: dict) -> dict:
    started = time.perf_counter()
    os.environ["AUTO_APPROVE_HUMAN"] = "true"
    thread_id = f"eval-{case['id']}-{uuid4().hex[:8]}"
    with get_checkpointer(Path("data/eval_checkpoints.db")) as checkpointer:
        state, finished, _ = run_until_pause_or_finish(
            case["input"],
            thread_id=thread_id,
            max_retry=0,
            checkpointer=checkpointer,
            verbose=False,
        )
        while not finished:
            state, finished, _ = resume_run(
                thread_id,
                {"approved": True, "comment": "eval"},
                checkpointer=checkpointer,
                verbose=False,
            )

    text = state.get("final_output") or ""
    flags = state.get("execution_flags")
    final_ok = bool(getattr(flags, "is_finished", False))
    terms_ok = _text_score(text, case.get("expected_terms", []))
    success = final_ok and terms_ok
    return {
        "id": case["id"],
        "type": "workflow",
        "success": success,
        "routing_ok": True,
        "skill_ok": True,
        "terms_ok": terms_ok,
        "sources_ok": True,
        "expected_worker": case.get("expected_worker"),
        "actual_worker": "workflow",
        "expected_skill": None,
        "actual_skill": None,
        "final_ok": final_ok,
        "task_count": len(state.get("task_list") or []),
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
    }


def evaluate(cases: list[dict], *, include_workflow: bool = True) -> dict:
    selected = [
        case
        for case in cases
        if include_workflow or case.get("type") != "workflow"
    ]
    details = [
        evaluate_workflow_case(case)
        if case.get("type") == "workflow"
        else evaluate_skill_case(case)
        for case in selected
    ]
    total = max(len(details), 1)
    latencies = [item["latency_ms"] for item in details]
    routing_items = [item for item in details if item["type"] == "skill"]
    return {
        "cases": len(details),
        "task_success_rate": round(sum(item["success"] for item in details) / total, 3),
        "routing_accuracy": round(
            sum(item["routing_ok"] for item in routing_items)
            / max(len(routing_items), 1),
            3,
        ),
        "skill_accuracy": round(
            sum(item["skill_ok"] for item in routing_items)
            / max(len(routing_items), 1),
            3,
        ),
        "term_coverage": round(sum(item["terms_ok"] for item in details) / total, 3),
        "source_coverage": round(
            sum(item["sources_ok"] for item in details) / total, 3
        ),
        "avg_latency_ms": round(sum(latencies) / total, 2),
        "p95_latency_ms": round(
            sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)], 2
        ),
        "failures": [item for item in details if not item["success"]],
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate routing, skills and workflows")
    parser.add_argument("--skip-workflow", action="store_true")
    parser.add_argument("--min-task-success", type=float, default=0.8)
    parser.add_argument("--min-routing-accuracy", type=float, default=0.8)
    parser.add_argument("--min-skill-accuracy", type=float, default=0.8)
    parser.add_argument("--min-term-coverage", type=float, default=0.8)
    args = parser.parse_args()

    cases = load_cases(Path(__file__).with_name("golden_tasks.jsonl"))
    result = evaluate(cases, include_workflow=not args.skip_workflow)
    print(json.dumps(result, ensure_ascii=False, indent=2))

    failures = []
    if result["task_success_rate"] < args.min_task_success:
        failures.append(f"task_success_rate={result['task_success_rate']}")
    if result["routing_accuracy"] < args.min_routing_accuracy:
        failures.append(f"routing_accuracy={result['routing_accuracy']}")
    if result["skill_accuracy"] < args.min_skill_accuracy:
        failures.append(f"skill_accuracy={result['skill_accuracy']}")
    if result["term_coverage"] < args.min_term_coverage:
        failures.append(f"term_coverage={result['term_coverage']}")
    if failures:
        raise SystemExit("evaluation below threshold: " + ", ".join(failures))


if __name__ == "__main__":
    main()
