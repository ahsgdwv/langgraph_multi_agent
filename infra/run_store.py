from __future__ import annotations

from uuid import uuid4

from infra.db import session_scope, tenant_session
from infra.models import RunEvent, RunRecord


def create_run_record(
    *,
    tenant_id: str,
    user_id: str,
    user_input: str,
    max_retry: int,
    auto_approve: bool,
    role: str = "viewer",
) -> RunRecord:
    run = RunRecord(
        id=uuid4().hex,
        tenant_id=tenant_id,
        user_id=user_id,
        role=role,
        status="queued",
        user_input=user_input,
        max_retry=max_retry,
        auto_approve=auto_approve,
        metrics_json={},
    )
    with tenant_session(tenant_id) as session:
        session.add(run)
        session.flush()
    return run


def get_run(tenant_id: str, run_id: str) -> RunRecord | None:
    with tenant_session(tenant_id) as session:
        return (
            session.query(RunRecord)
            .filter(RunRecord.id == run_id, RunRecord.tenant_id == tenant_id)
            .first()
        )


def get_run_by_id(run_id: str) -> RunRecord | None:
    with session_scope() as session:
        return session.get(RunRecord, run_id)


def list_runs(tenant_id: str, limit: int = 50) -> list[RunRecord]:
    with tenant_session(tenant_id) as session:
        return (
            session.query(RunRecord)
            .filter(RunRecord.tenant_id == tenant_id)
            .order_by(RunRecord.created_at.desc())
            .limit(limit)
            .all()
        )


def update_run(
    run_id: str,
    *,
    tenant_id: str | None = None,
    **fields,
) -> RunRecord | None:
    context = tenant_session(tenant_id) if tenant_id else session_scope()
    with context as session:
        run = session.get(RunRecord, run_id)
        if run is None:
            return None
        for key, value in fields.items():
            if hasattr(run, key):
                setattr(run, key, value)
        session.flush()
        return run


def add_event(
    *,
    run_id: str,
    tenant_id: str,
    event_type: str,
    payload: dict | None = None,
) -> RunEvent:
    event = RunEvent(
        run_id=run_id,
        tenant_id=tenant_id,
        event_type=event_type,
        payload_json=payload or {},
    )
    with tenant_session(tenant_id) as session:
        session.add(event)
        session.flush()
    return event


def list_events(tenant_id: str, run_id: str, limit: int = 200) -> list[RunEvent]:
    with tenant_session(tenant_id) as session:
        return (
            session.query(RunEvent)
            .filter(RunEvent.tenant_id == tenant_id, RunEvent.run_id == run_id)
            .order_by(RunEvent.id.asc())
            .limit(limit)
            .all()
        )


def serialize_run(run: RunRecord) -> dict:
    return {
        "run_id": run.id,
        "tenant_id": run.tenant_id,
        "user_id": run.user_id,
        "role": run.role,
        "status": run.status,
        "user_input": run.user_input,
        "final_output": run.final_output,
        "metrics": run.metrics_json or {},
        "interrupt": run.interrupt_json,
        "error": run.error,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "updated_at": run.updated_at.isoformat() if run.updated_at else None,
    }


def serialize_event(event: RunEvent) -> dict:
    return {
        "event_id": event.id,
        "run_id": event.run_id,
        "event_type": event.event_type,
        "payload": event.payload_json or {},
        "created_at": event.created_at.isoformat() if event.created_at else None,
    }
