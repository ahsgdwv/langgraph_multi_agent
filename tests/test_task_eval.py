from __future__ import annotations

from pathlib import Path

from evals.task_eval import evaluate, load_cases


def test_end_to_end_task_eval_quality_gates():
    cases = load_cases(Path("evals/golden_tasks.jsonl"))
    result = evaluate(cases, include_workflow=False)
    assert result["task_success_rate"] >= 0.95
    assert result["routing_accuracy"] >= 0.95
    assert result["skill_accuracy"] >= 0.95
    assert result["term_coverage"] >= 0.9
