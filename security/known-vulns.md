# Known dependency exceptions

`pip-audit` currently reports four unique advisories against the pinned
`chromadb==1.5.9`:

- PYSEC-2026-311 / CVE-2026-45829: remote model loading can execute code when a
  Chroma **server** is configured with `trust_remote_code`.
- PYSEC-2026-3813 / CVE-2026-45830: missing authorization checks in the Chroma
  server API.
- PYSEC-2026-3814 / CVE-2026-45833: authenticated code injection through a
  malicious model repository.
- PYSEC-2026-3815 / CVE-2026-45831: tenant/database scope is not checked by
  Chroma's SimpleRBAC provider.

## Why the CI exception exists

This application currently uses the embedded Chroma client only. It does not
expose Chroma's `/api/v2` server endpoints, and it does not load remote model
repositories. The server-side advisories therefore do not match the current
deployment surface.

The exception is deliberately narrow: new advisories still fail the security
job.

## Required migration before multi-tenant production

Do not run an unauthenticated Chroma server for multiple tenants. Before using
Chroma as a shared service, use either:

1. PostgreSQL + pgvector with row-level tenant scoping, or
2. an isolated Chroma deployment with authentication, network policy, and
   per-tenant collections.

Track upstream fixed releases and remove these exceptions as soon as a patched
version is available.
