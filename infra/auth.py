from __future__ import annotations

import hmac
from dataclasses import dataclass
from typing import Callable, Literal

from fastapi import Depends, HTTPException, Request

from infra.settings import get_settings

Role = Literal["admin", "analyst", "viewer"]
_ROLES = {"admin", "analyst", "viewer"}


@dataclass(frozen=True)
class Principal:
    tenant_id: str
    user_id: str
    role: Role


def _extract_api_key(request: Request) -> str:
    key = request.headers.get("X-API-Key", "").strip()
    if key:
        return key
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def get_principal(request: Request) -> Principal:
    settings = get_settings()
    if settings.auth_mode == "dev":
        tenant_id = request.headers.get("X-Tenant-ID", "demo").strip() or "demo"
        user_id = request.headers.get("X-User-ID", "dev").strip() or "dev"
        return Principal(tenant_id=tenant_id, user_id=user_id, role="admin")

    key = _extract_api_key(request)
    if not key:
        raise HTTPException(
            status_code=401,
            detail="缺少 API Key",
            headers={"WWW-Authenticate": "Bearer"},
        )
    for candidate, metadata in settings.api_keys.items():
        if not hmac.compare_digest(candidate, key):
            continue
        role = str(metadata.get("role", "viewer"))
        if role not in _ROLES:
            raise HTTPException(status_code=403, detail="API Key 角色无效")
        tenant_id = str(metadata.get("tenant_id", "")).strip()
        if not tenant_id:
            raise HTTPException(status_code=403, detail="API Key 未绑定租户")
        return Principal(
            tenant_id=tenant_id,
            user_id=str(metadata.get("user_id", "api")),
            role=role,  # type: ignore[arg-type]
        )
    raise HTTPException(status_code=401, detail="API Key 无效")


def require_roles(*roles: Role) -> Callable[[Principal], Principal]:
    allowed = set(roles)

    def dependency(principal: Principal = Depends(get_principal)) -> Principal:
        if principal.role not in allowed:
            raise HTTPException(status_code=403, detail="权限不足")
        return principal

    return dependency
