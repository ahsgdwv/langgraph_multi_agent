# pgvector backend

Set:

```env
VECTOR_BACKEND=pgvector
EMBEDDING_DIMENSIONS=384
DATABASE_URL=postgresql+psycopg://fmcg:change-me@db:5432/fmcg
```

Apply `security/pgvector.sql` after the application tables exist. The table is
tenant-scoped, with a `shared` tenant for the demo document library. Real
tenant-owned documents should be ingested with the tenant id instead of
`SHARED_DOCUMENT_TENANT`.

The HNSW cosine index is created for 384-dimensional embeddings. If you switch
to a different embedding model, update both `EMBEDDING_DIMENSIONS` and the
`vector(n)` column/index in the migration.
