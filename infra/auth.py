from __future__ import annotations

import hmac
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

import jwt
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


def _extract_bearer_token(request: Request) -> str:
    authorization = request.headers.get("Authorization", "")
    if authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


@lru_cache
def _jwks_client(jwks_url: str) -> jwt.PyJWKClient:
    return jwt.PyJWKClient(jwks_url)


def _decode_oidc_token(token: str, settings) -> dict:
    if not settings.oidc_jwks_url:
        raise HTTPException(status_code=500, detail="OIDC_JWKS_URL 未配置")
    try:
        signing_key = _jwks_client(settings.oidc_jwks_url).get_signing_key_from_jwt(token)
        return jwt.decode(
            token,
            signing_key.key,
            algorithms=settings.oidc_algorithm_list,
            audience=settings.oidc_audience or None,
            issuer=settings.oidc_issuer or None,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail="OIDC Token 无效") from exc


def _principal_from_claims(claims: dict, settings) -> Principal:
    raw_tenant = claims.get(settings.oidc_tenant_claim) or claims.get("tenant_id")
    if isinstance(raw_tenant, list):
        raw_tenant = raw_tenant[0] if raw_tenant else None
    tenant_id = str(raw_tenant or "").strip()
    if not tenant_id:
        raise HTTPException(status_code=403, detail="OIDC Token 未绑定租户")

    raw_role = claims.get(settings.oidc_role_claim) or claims.get("roles") or "viewer"
    if isinstance(raw_role, list):
        raw_role = next((item for item in raw_role if item in _ROLES), "viewer")
    role = str(raw_role)
    if role not in _ROLES:
        raise HTTPException(status_code=403, detail="OIDC 角色无效")

    user_id = str(claims.get("sub") or claims.get("email") or "oidc-user")
    return Principal(tenant_id=tenant_id, user_id=user_id, role=role)  # type: ignore[arg-type]


def get_principal(request: Request) -> Principal:
    settings = get_settings()
    if settings.auth_mode == "dev":
        tenant_id = request.headers.get("X-Tenant-ID", "demo").strip() or "demo"
        user_id = request.headers.get("X-User-ID", "dev").strip() or "dev"
        return Principal(tenant_id=tenant_id, user_id=user_id, role="admin")

    if settings.auth_mode == "oidc":
        token = _extract_bearer_token(request)
        if not token:
            raise HTTPException(
                status_code=401,
                detail="缺少 OIDC Bearer Token",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return _principal_from_claims(_decode_oidc_token(token, settings), settings)

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
