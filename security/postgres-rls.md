# PostgreSQL Row-Level Security

`postgres_rls.sql` enables tenant isolation at the database layer for:

- `runs`
- `run_events`
- `jobs`
- `audit_log`

The application sets `app.tenant_id` inside `infra.db.tenant_session()` for
request-scoped writes and reads.

Use a separate worker database role with `BYPASSRLS` for:

- claiming jobs across tenants
- loading a run by id
- finalizing a run and job in one transaction

Do not grant `BYPASSRLS` to the API role. Apply the SQL with the table owner
account, then verify with two tenants that each can only see its own runs and
audit entries.
