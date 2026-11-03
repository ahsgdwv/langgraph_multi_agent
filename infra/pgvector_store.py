from __future__ import annotations

from sqlalchemy import text

from infra.db import get_engine, tenant_session
from infra.settings import get_settings


def _vector_literal(values: list[float]) -> str:
    return "[" + ",".join(f"{value:.8f}" for value in values) + "]"


def ensure_pgvector_schema() -> None:
    dimensions = get_settings().embedding_dimensions
    with get_engine().begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        connection.execute(
            text(
                f"""
                CREATE TABLE IF NOT EXISTS document_chunks (
                    id BIGSERIAL PRIMARY KEY,
                    tenant_id VARCHAR(64) NOT NULL,
                    source VARCHAR(255) NOT NULL,
                    content TEXT NOT NULL,
                    embedding vector({dimensions}) NOT NULL
                )
                """
            )
        )
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx "
                "ON document_chunks USING hnsw (embedding vector_cosine_ops)"
            )
        )


def replace_chunks(chunks: list[dict], *, tenant_id: str) -> None:
    ensure_pgvector_schema()
    with tenant_session(tenant_id) as session:
        session.execute(
            text("DELETE FROM document_chunks WHERE tenant_id = :tenant_id"),
            {"tenant_id": tenant_id},
        )
        for chunk in chunks:
            session.execute(
                text(
                    """
                    INSERT INTO document_chunks (tenant_id, source, content, embedding)
                    VALUES (:tenant_id, :source, :content, CAST(:embedding AS vector))
                    """
                ),
                {
                    "tenant_id": tenant_id,
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "embedding": _vector_literal(chunk["embedding"]),
                },
            )


def search_chunks(
    embedding: list[float],
    *,
    tenant_id: str,
    k: int,
    allowed_sources: list[str] | None = None,
) -> list[dict]:
    ensure_pgvector_schema()
    params: dict = {
        "tenant_id": tenant_id,
        "shared_tenant": get_settings().shared_document_tenant,
        "embedding": _vector_literal(embedding),
        "limit": k,
    }
    source_clause = ""
    if allowed_sources is not None:
        source_clause = "AND source = ANY(:sources)"
        params["sources"] = allowed_sources
    with tenant_session(tenant_id) as session:
        rows = session.execute(
            text(
                f"""
                SELECT source, content,
                       1 - (embedding <=> CAST(:embedding AS vector)) AS score
                FROM document_chunks
                WHERE tenant_id IN (:tenant_id, :shared_tenant)
                {source_clause}
                ORDER BY embedding <=> CAST(:embedding AS vector)
                LIMIT :limit
                """
            ),
            params,
        ).mappings()
        return [
            {
                "source": row["source"],
                "content": row["content"],
                "score": float(row["score"]),
            }
            for row in rows
        ]
