-- Apply as the database owner after creating the application tables.
-- API/request connections must run:
--   SELECT set_config('app.tenant_id', '<tenant-id>', true);
-- Worker connections should use a dedicated role with BYPASSRLS.

ALTER TABLE runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE run_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE audit_log ENABLE ROW LEVEL SECURITY;
ALTER TABLE jobs ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS tenant_isolation_runs ON runs;
CREATE POLICY tenant_isolation_runs ON runs
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS tenant_isolation_run_events ON run_events;
CREATE POLICY tenant_isolation_run_events ON run_events
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS tenant_isolation_audit_log ON audit_log;
CREATE POLICY tenant_isolation_audit_log ON audit_log
    USING (tenant_id = current_setting('app.tenant_id', true));

DROP POLICY IF EXISTS tenant_isolation_jobs ON jobs;
CREATE POLICY tenant_isolation_jobs ON jobs
    USING (
        EXISTS (
            SELECT 1
            FROM runs
            WHERE runs.id = jobs.run_id
              AND runs.tenant_id = current_setting('app.tenant_id', true)
        )
    );

-- The worker role needs cross-tenant access to claim queued jobs:
-- CREATE ROLE fmcg_worker BYPASSRLS LOGIN;
