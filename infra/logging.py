from __future__ import annotations

import json
import logging
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any

from infra.settings import get_settings

_request_id: ContextVar[str] = ContextVar("request_id", default="")
_run_id: ContextVar[str] = ContextVar("run_id", default="")
_tenant_id: ContextVar[str] = ContextVar("tenant_id", default="")
_task_id: ContextVar[str] = ContextVar("task_id", default="")

_CONTEXT = {
    "request_id": _request_id,
    "run_id": _run_id,
    "tenant_id": _tenant_id,
    "task_id": _task_id,
}


def bind_context(**values: Any) -> dict[str, Any]:
    tokens: dict[str, Any] = {}
    for key, value in values.items():
        variable = _CONTEXT.get(key)
        if variable is not None:
            tokens[key] = variable.set(str(value))
    return tokens


def clear_context(tokens: dict[str, Any]) -> None:
    for key, token in tokens.items():
        _CONTEXT[key].reset(token)


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        for key, variable in _CONTEXT.items():
            value = variable.get()
            if value:
                payload[key] = value
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, ensure_ascii=False)


def configure_logging(level: str | None = None, json_logs: bool | None = None) -> None:
    settings = get_settings()
    handler = logging.StreamHandler()
    if json_logs if json_logs is not None else settings.log_json:
        handler.setFormatter(JsonFormatter())
    else:
        handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
        )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(str(level or settings.log_level).upper())


logger = logging.getLogger("fmcg")
