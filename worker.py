from __future__ import annotations

import time
from threading import Event, Thread
from uuid import uuid4

from infra.audit_store import append_audit
from infra.db import init_db
from infra.job_queue import (
    claim_job,
    complete_job,
    fail_job,
    heartbeat_job,
    reset_stale_jobs,
)
from infra.logging import bind_context, clear_context, configure_logging, logger
from infra.run_store import add_event, get_run_by_id, update_run
from infra.settings import get_settings
from main import get_checkpointer, resume_run, run_until_pause_or_finish


def _metrics_dict(state: dict) -> dict:
    metrics = state.get("run_metrics")
    if hasattr(metrics, "model_dump"):
        return metrics.model_dump()
    return dict(metrics or {})


def process_job(job: dict, worker_id: str) -> None:
    run = get_run_by_id(job["run_id"])
    if run is None:
        complete_job(job["job_id"])
        logger.warning("job_run_missing job_id=%s run_id=%s", job["job_id"], job["run_id"])
        return
    if run.status == "cancelled":
        complete_job(job["job_id"])
        logger.info("job_cancelled run_id=%s", run.id)
        return

    tokens = bind_context(run_id=run.id, tenant_id=run.tenant_id, task_id=job["job_id"])
    update_run(run.id, status="running", error=None)
    add_event(
        run_id=run.id,
        tenant_id=run.tenant_id,
        event_type="started",
        payload={"job_id": job["job_id"], "worker_id": worker_id},
    )
    append_audit(
        tenant_id=run.tenant_id,
        actor_id=f"worker:{worker_id}",
        action="run.start",
        resource_type="run",
        resource_id=run.id,
        details={"job_id": job["job_id"], "action": job["action"]},
    )
    stop_heartbeat = Event()

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(30):
            heartbeat_job(job["job_id"], worker_id)

    heartbeat = Thread(target=heartbeat_loop, daemon=True)
    heartbeat.start()
    try:
        with get_checkpointer() as checkpointer:
            if job["action"] == "resume":
                state, finished, interrupt_payload = resume_run(
                    run.id,
                    job["payload"],
                    checkpointer=checkpointer,
                    verbose=False,
                )
            else:
                state, finished, interrupt_payload = run_until_pause_or_finish(
                    run.user_input,
                    thread_id=run.id,
                    max_retry=run.max_retry,
                    auto_approve_human=run.auto_approve,
                    tenant_id=run.tenant_id,
                    user_role=run.role,
                    checkpointer=checkpointer,
                    verbose=False,
                )

        status = "succeeded" if finished else "interrupted"
        add_event(
            run_id=run.id,
            tenant_id=run.tenant_id,
            event_type=status,
            payload={"job_id": job["job_id"]},
        )
        append_audit(
            tenant_id=run.tenant_id,
            actor_id=f"worker:{worker_id}",
            action="run.finish",
            resource_type="run",
            resource_id=run.id,
            details={"status": status, "job_id": job["job_id"]},
        )
        complete_job(
            job["job_id"],
            run_id=run.id,
            run_fields={
                "status": status,
                "final_output": state.get("final_output") or "",
                "metrics_json": _metrics_dict(state),
                "interrupt_json": interrupt_payload,
                "error": None,
            },
        )
        logger.info("job_finished run_id=%s status=%s", run.id, status)
    except Exception as exc:
        retry = fail_job(
            job["job_id"],
            str(exc),
            retry=True,
            run_id=run.id,
        )
        add_event(
            run_id=run.id,
            tenant_id=run.tenant_id,
            event_type="retry_scheduled" if retry else "failed",
            payload={"job_id": job["job_id"], "error": str(exc)},
        )
        append_audit(
            tenant_id=run.tenant_id,
            actor_id=f"worker:{worker_id}",
            action="run.retry" if retry else "run.failure",
            resource_type="run",
            resource_id=run.id,
            details={"error": str(exc)[:500], "job_id": job["job_id"]},
        )
        logger.exception("job_failed run_id=%s retry=%s", run.id, retry)
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=1)
        clear_context(tokens)


def run_once(worker_id: str = "worker-once") -> bool:
    job = claim_job(worker_id)
    if job is None:
        return False
    process_job(job, worker_id)
    return True


def run_forever() -> None:
    configure_logging()
    init_db()
    settings = get_settings()
    worker_id = settings.worker_id or f"worker-{uuid4().hex[:8]}"
    reset_stale_jobs()
    logger.info("worker_started worker_id=%s", worker_id)
    try:
        while True:
            if not run_once(worker_id):
                time.sleep(settings.job_poll_seconds)
    except KeyboardInterrupt:
        logger.info("worker_stopped worker_id=%s", worker_id)


if __name__ == "__main__":
    run_forever()
