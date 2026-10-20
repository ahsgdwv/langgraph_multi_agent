from __future__ import annotations

import json

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from starlette.requests import Request

from infra.auth import get_principal
from infra.logging import JsonFormatter, bind_context, clear_context
from infra.settings import get_settings


def _request(headers: dict[str, str]) -> Request:
    return Request(
        {
            "type": "http",
            "method": "GET",
            "path": "/",
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers.items()],
        }
    )


def test_dev_auth_uses_tenant_header():
    principal = get_principal(
        _request({"X-Tenant-ID": "acme", "X-User-ID": "alice"})
    )
    assert principal.tenant_id == "acme"
    assert principal.user_id == "alice"
    assert principal.role == "admin"


def test_api_key_auth_and_invalid_key(monkeypatch):
    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv(
        "API_KEYS_JSON",
        json.dumps(
            {
                "secret-key": {
                    "tenant_id": "acme",
                    "user_id": "analyst-1",
                    "role": "analyst",
                }
            }
        ),
    )
    get_settings.cache_clear()
    try:
        principal = get_principal(_request({"X-API-Key": "secret-key"}))
        assert principal.tenant_id == "acme"
        assert principal.role == "analyst"
        with pytest.raises(HTTPException) as exc:
            get_principal(_request({"X-API-Key": "wrong"}))
        assert exc.value.status_code == 401
    finally:
        monkeypatch.delenv("AUTH_MODE", raising=False)
        monkeypatch.delenv("API_KEYS_JSON", raising=False)
        get_settings.cache_clear()


def test_json_logging_includes_run_context():
    tokens = bind_context(run_id="run-1", tenant_id="acme")
    try:
        record = __import__("logging").LogRecord(
            "fmcg.test", __import__("logging").INFO, __file__, 1, "hello", None, None
        )
        payload = json.loads(JsonFormatter().format(record))
        assert payload["message"] == "hello"
        assert payload["run_id"] == "run-1"
        assert payload["tenant_id"] == "acme"
    finally:
        clear_context(tokens)


def test_async_run_queue_and_worker():
    from api import app
    from worker import run_once

    client = TestClient(app)
    created = client.post(
        "/v1/runs",
        json={
            "user_input": "统计渠道销量、输出SKU排行",
            "max_retry": 0,
            "auto_approve": True,
        },
        headers={"X-Tenant-ID": "acme"},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]
    assert created.json()["status"] == "queued"

    assert run_once("test-worker") is True
    detail = client.get(f"/v1/runs/{run_id}", headers={"X-Tenant-ID": "acme"})
    assert detail.status_code == 200
    body = detail.json()
    assert body["status"] in ("succeeded", "interrupted")
    assert body["events"]


def test_tenant_isolation_for_runs():
    from api import app

    client = TestClient(app)
    created = client.post(
        "/v1/runs",
        json={"user_input": "统计渠道销量", "max_retry": 0, "auto_approve": True},
        headers={"X-Tenant-ID": "tenant-a"},
    )
    assert created.status_code == 200
    run_id = created.json()["run_id"]
    hidden = client.get(f"/v1/runs/{run_id}", headers={"X-Tenant-ID": "tenant-b"})
    assert hidden.status_code == 404


def test_viewer_role_cannot_create_run(monkeypatch):
    from api import app

    monkeypatch.setenv("AUTH_MODE", "api_key")
    monkeypatch.setenv(
        "API_KEYS_JSON",
        json.dumps(
            {
                "viewer-key": {
                    "tenant_id": "acme",
                    "user_id": "viewer-1",
                    "role": "viewer",
                }
            }
        ),
    )
    get_settings.cache_clear()
    try:
        response = TestClient(app).post(
            "/v1/runs",
            json={"user_input": "统计渠道销量"},
            headers={"X-API-Key": "viewer-key"},
        )
        assert response.status_code == 403
    finally:
        monkeypatch.delenv("AUTH_MODE", raising=False)
        monkeypatch.delenv("API_KEYS_JSON", raising=False)
        get_settings.cache_clear()
