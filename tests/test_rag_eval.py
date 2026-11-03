from __future__ import annotations

from evals.rag_eval import evaluate, load_cases
from rag_store import get_vectorstore, ingest_documents


def test_rag_eval_hits_threshold():
    result = ingest_documents(force=True)
    assert result["ok"] is True
    assert get_vectorstore()._collection.count() == result["chunks"]

    cases = load_cases(__import__("pathlib").Path("evals/golden_rag.jsonl"))
    metrics = evaluate(cases, k=4)
    assert metrics["hit_rate"] >= 0.8
    assert metrics["term_hit_rate"] >= 0.8


def test_document_acl_blocks_unknown_tenants():
    from rag_store import retrieve_context

    ingest_documents(force=True)
    allowed = retrieve_context("陈列费用政策", k=2, tenant_id="acme", role="viewer")
    assert allowed
    blocked = retrieve_context("陈列费用政策", k=2, tenant_id="unknown", role="viewer")
    assert blocked == []
