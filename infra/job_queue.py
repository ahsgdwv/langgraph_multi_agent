from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import uuid4

from infra.db import get_engine, session_scope
from infra.models import JobRecord, RunRecord
from infra.settings import get_settings


def _now() -> datetime:
    return datetime.now(timezone.utc)


def enqueue_job(
    *,
    run_id: str,
    action: str = "start",
    payload: dict | None = None,
    max_attempts: int | None = None,
) -> JobRecord:
    with session_scope() as session:
        existing = (
            session.query(JobRecord)
            .filter(
                JobRecord.run_id == run_id,
                JobRecord.action == action,
                JobRecord.status.in_(("pending", "running")),
            )
            .first()
        )
        if existing is not None:
            return existing

    job = JobRecord(
        id=uuid4().hex,
        run_id=run_id,
        action=action,
        payload_json=payload or {},
        status="pending",
        attempts=0,
        max_attempts=max_attempts or get_settings().job_max_attempts,
        available_at=_now(),
    )
    with session_scope() as session:
        session.add(job)
        session.flush()
    return job


def claim_job(worker_id: str) -> dict | None:
    now = _now()
    with session_scope() as session:
        query = (
            session.query(JobRecord)
            .filter(JobRecord.status == "pending", JobRecord.available_at <= now)
            .order_by(JobRecord.created_at.asc())
        )
        if get_engine().dialect.name == "postgresql":
            query = query.with_for_update(skip_locked=True)
        job = query.first()
        if job is None:
            return None
        job.status = "running"
        job.worker_id = worker_id
        job.locked_at = now
        job.attempts += 1
        session.flush()
        return serialize_job(job)


def heartbeat_job(job_id: str, worker_id: str) -> bool:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None or job.status != "running" or job.worker_id != worker_id:
            return False
        job.locked_at = _now()
        return True


def complete_job(
    job_id: str,
    *,
    run_id: str | None = None,
    run_fields: dict | None = None,
) -> None:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            return
        job.status = "done"
        job.locked_at = None
        job.error = None
        if run_id:
            run = session.get(RunRecord, run_id)
            if run is not None:
                for key, value in (run_fields or {}).items():
                    if hasattr(run, key):
                        setattr(run, key, value)


def fail_job(
    job_id: str,
    error: str,
    *,
    retry: bool = True,
    run_id: str | None = None,
) -> bool:
    with session_scope() as session:
        job = session.get(JobRecord, job_id)
        if job is None:
            return False
        job.error = error[:4000]
        job.locked_at = None
        job.worker_id = None
        if retry and job.attempts < job.max_attempts:
            delay = min(60, 2 ** max(job.attempts, 0))
            job.status = "pending"
            job.available_at = _now() + timedelta(seconds=delay)
            should_retry = True
        else:
            job.status = "failed"
            should_retry = False
        if run_id:
            run = session.get(RunRecord, run_id)
            if run is not None:
                run.status = "queued" if should_retry else "failed"
                run.error = error[:4000]
        return should_retry


def cancel_jobs_for_run(run_id: str) -> None:
    with session_scope() as session:
        jobs = (
            session.query(JobRecord)
            .filter(JobRecord.run_id == run_id)
            .filter(JobRecord.status.in_(("pending", "running")))
            .all()
        )
        for job in jobs:
            job.status = "cancelled"
            job.locked_at = None


def reset_stale_jobs(timeout_seconds: int = 300) -> int:
    cutoff = _now() - timedelta(seconds=timeout_seconds)
    with session_scope() as session:
        jobs = (
            session.query(JobRecord)
            .filter(JobRecord.status == "running", JobRecord.locked_at < cutoff)
            .all()
        )
        for job in jobs:
            job.status = "pending"
            job.available_at = _now()
            job.locked_at = None
            job.worker_id = None
        return len(jobs)


def serialize_job(job: JobRecord) -> dict:
    return {
        "job_id": job.id,
        "run_id": job.run_id,
        "action": job.action,
        "payload": job.payload_json or {},
        "status": job.status,
        "attempts": job.attempts,
        "max_attempts": job.max_attempts,
    }
