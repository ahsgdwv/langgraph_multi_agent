-- Requires PostgreSQL with the pgvector extension available.
-- Keep the vector dimension aligned with EMBEDDING_DIMENSIONS.

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGSERIAL PRIMARY KEY,
    tenant_id VARCHAR(64) NOT NULL,
    source VARCHAR(255) NOT NULL,
    content TEXT NOT NULL,
    embedding vector(384) NOT NULL
);

CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
    ON document_chunks USING hnsw (embedding vector_cosine_ops);

ALTER TABLE document_chunks ENABLE ROW LEVEL SECURITY;
ALTER TABLE document_chunks FORCE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_document_chunks ON document_chunks;
CREATE POLICY tenant_isolation_document_chunks ON document_chunks
    FOR ALL
    USING (
        tenant_id = current_setting('app.tenant_id', true)
        OR tenant_id = 'shared'
    )
    WITH CHECK (
        tenant_id = current_setting('app.tenant_id', true)
        OR tenant_id = 'shared'
    );
