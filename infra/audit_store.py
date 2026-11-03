from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from infra.db import get_engine, tenant_session
from infra.models import AuditRecord


def _iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.isoformat()


def _hash_entry(
    *,
    tenant_id: str,
    actor_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict,
    occurred_at: str,
    prev_hash: str,
) -> str:
    body = {
        "tenant_id": tenant_id,
        "actor_id": actor_id,
        "action": action,
        "resource_type": resource_type,
        "resource_id": resource_id,
        "details": details,
        "occurred_at": occurred_at,
        "prev_hash": prev_hash,
    }
    raw = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def append_audit(
    *,
    tenant_id: str,
    actor_id: str,
    action: str,
    resource_type: str,
    resource_id: str,
    details: dict | None = None,
) -> AuditRecord:
    details = details or {}
    for _ in range(3):
        try:
            with tenant_session(tenant_id) as session:
                query = (
                    session.query(AuditRecord)
                    .filter(AuditRecord.tenant_id == tenant_id)
                    .order_by(AuditRecord.id.desc())
                )
                if get_engine().dialect.name == "postgresql":
                    query = query.with_for_update()
                previous = query.first()
                prev_hash = previous.entry_hash if previous else "GENESIS"
                occurred_at = datetime.now(timezone.utc)
                entry_hash = _hash_entry(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details=details,
                    occurred_at=_iso(occurred_at),
                    prev_hash=prev_hash,
                )
                record = AuditRecord(
                    tenant_id=tenant_id,
                    actor_id=actor_id,
                    action=action,
                    resource_type=resource_type,
                    resource_id=resource_id,
                    details_json=details,
                    prev_hash=prev_hash,
                    entry_hash=entry_hash,
                    occurred_at=occurred_at,
                )
                session.add(record)
                session.flush()
                return record
        except IntegrityError:
            continue
    raise RuntimeError("审计链写入冲突，请重试")


def list_audit(tenant_id: str, limit: int = 200) -> list[AuditRecord]:
    with tenant_session(tenant_id) as session:
        return (
            session.query(AuditRecord)
            .filter(AuditRecord.tenant_id == tenant_id)
            .order_by(AuditRecord.id.desc())
            .limit(limit)
            .all()
        )


def verify_audit_chain(tenant_id: str) -> dict:
    with tenant_session(tenant_id) as session:
        records = (
            session.query(AuditRecord)
            .filter(AuditRecord.tenant_id == tenant_id)
            .order_by(AuditRecord.id.asc())
            .all()
        )
    previous = "GENESIS"
    for index, record in enumerate(records, 1):
        expected = _hash_entry(
            tenant_id=record.tenant_id,
            actor_id=record.actor_id,
            action=record.action,
            resource_type=record.resource_type,
            resource_id=record.resource_id,
            details=record.details_json or {},
            occurred_at=_iso(record.occurred_at),
            prev_hash=record.prev_hash,
        )
        if record.prev_hash != previous or record.entry_hash != expected:
            return {"valid": False, "count": len(records), "broken_at": index}
        previous = record.entry_hash
    return {"valid": True, "count": len(records), "broken_at": None}


def serialize_audit(record: AuditRecord) -> dict:
    return {
        "id": record.id,
        "tenant_id": record.tenant_id,
        "actor_id": record.actor_id,
        "action": record.action,
        "resource_type": record.resource_type,
        "resource_id": record.resource_id,
        "details": record.details_json or {},
        "prev_hash": record.prev_hash,
        "entry_hash": record.entry_hash,
        "occurred_at": _iso(record.occurred_at) if record.occurred_at else None,
    }
