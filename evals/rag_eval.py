from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_store import ingest_documents, retrieve_context


def load_cases(path: Path) -> list[dict]:
    cases: list[dict] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            cases.append(json.loads(line))
    return cases


def evaluate(cases: list[dict], *, k: int = 4) -> dict:
    hits = 0
    reciprocal_rank = 0.0
    term_hits = 0
    details: list[dict] = []

    for case in cases:
        docs = retrieve_context(case["query"], k=k)
        sources = [doc.source for doc in docs]
        expected_sources = case.get("expected_sources", [])
        rank = next(
            (
                index + 1
                for index, source in enumerate(sources)
                if source in expected_sources
            ),
            None,
        )
        if rank is not None:
            hits += 1
            reciprocal_rank += 1.0 / rank

        text = "\n".join(doc.content for doc in docs).lower()
        expected_terms = case.get("expected_terms", [])
        terms_ok = all(str(term).lower() in text for term in expected_terms)
        if terms_ok:
            term_hits += 1

        details.append(
            {
                "query": case["query"],
                "sources": sources,
                "source_hit": rank is not None,
                "rank": rank,
                "terms_ok": terms_ok,
            }
        )

    total = max(len(cases), 1)
    return {
        "cases": len(cases),
        "k": k,
        "hit_rate": round(hits / total, 3),
        "mrr": round(reciprocal_rank / total, 3),
        "term_hit_rate": round(term_hits / total, 3),
        "details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate RAG retrieval on a golden set")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--min-hit-rate", type=float, default=0.8)
    args = parser.parse_args()

    if args.rebuild:
        ingest_documents(force=True)
    cases = load_cases(Path(__file__).with_name("golden_rag.jsonl"))
    result = evaluate(cases, k=args.k)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["hit_rate"] < args.min_hit_rate:
        raise SystemExit(
            f"hit_rate {result['hit_rate']} < {args.min_hit_rate}"
        )


if __name__ == "__main__":
    main()
