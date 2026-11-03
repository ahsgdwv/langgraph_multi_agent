from __future__ import annotations

import os
from pathlib import Path

import psycopg
import pytest

POSTGRES_DSN = os.getenv("POSTGRES_DSN", "")

pytestmark = pytest.mark.skipif(
    not POSTGRES_DSN,
    reason="POSTGRES_DSN is not configured",
)


def _apply_sql(path: Path) -> None:
    with psycopg.connect(POSTGRES_DSN, autocommit=True) as connection:
        for statement in path.read_text(encoding="utf-8").split(";"):
            if statement.strip():
                connection.execute(statement)


def test_postgres_rls_and_pgvector(monkeypatch):
    monkeypatch.setenv("DATABASE_URL", POSTGRES_DSN.replace("postgresql://", "postgresql+psycopg://"))
    monkeypatch.setenv("VECTOR_BACKEND", "pgvector")
    monkeypatch.setenv("EMBEDDING_DIMENSIONS", "384")

    from infra.settings import get_settings

    get_settings.cache_clear()
    from infra.db import init_db, reset_engine

    reset_engine()
    init_db()
    _apply_sql(Path("security/postgres_rls.sql"))
    _apply_sql(Path("security/pgvector.sql"))

    from infra.pgvector_store import replace_chunks, search_chunks
    from infra.run_store import create_run_record, get_run

    run = create_run_record(
        tenant_id="rls-tenant-a",
        user_id="alice",
        user_input="统计渠道销量",
        max_retry=0,
        auto_approve=True,
        role="analyst",
    )
    assert get_run("rls-tenant-a", run.id) is not None
    assert get_run("rls-tenant-b", run.id) is None

    replace_chunks(
        [
            {
                "source": "policy.md",
                "content": "便利店陈列费不超过月销售额的 8%",
                "embedding": [0.0] * 384,
            }
        ],
        tenant_id="shared",
    )
    rows = search_chunks(
        [0.0] * 384,
        tenant_id="rls-tenant-a",
        k=2,
    )
    assert rows
    assert rows[0]["source"] == "policy.md"

    reset_engine()
    get_settings.cache_clear()
