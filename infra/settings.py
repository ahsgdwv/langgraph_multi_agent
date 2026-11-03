from __future__ import annotations

import json
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Environment variables override .env values."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_name: str = "fmcg-agent"
    environment: Literal["dev", "staging", "prod"] = "dev"

    database_url: str = "sqlite:///data/app.db"
    checkpoint_backend: Literal["sqlite", "postgres"] = "sqlite"
    checkpoint_sqlite_path: str = "data/checkpoints.db"
    postgres_dsn: str = ""

    auth_mode: Literal["dev", "api_key", "oidc"] = "dev"
    api_keys_json: str = ""
    oidc_issuer: str = ""
    oidc_audience: str = ""
    oidc_jwks_url: str = ""
    oidc_algorithms: str = "RS256,ES256"
    oidc_tenant_claim: str = "tenant_id"
    oidc_role_claim: str = "role"

    log_level: str = "INFO"
    log_json: bool = True
    job_poll_seconds: float = Field(default=1.0, ge=0.1)
    job_max_attempts: int = Field(default=3, ge=1)
    worker_id: str = ""

    embedding_provider: Literal["auto", "hash", "openai"] = "auto"
    embedding_model: str = "text-embedding-3-small"

    @property
    def api_keys(self) -> dict[str, dict[str, str]]:
        raw = self.api_keys_json.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError("API_KEYS_JSON 必须是合法 JSON") from exc
        if not isinstance(parsed, dict):
            raise ValueError("API_KEYS_JSON 必须是 key -> metadata 的对象")
        return {
            str(key): dict(value)
            for key, value in parsed.items()
            if isinstance(value, dict)
        }

    @property
    def oidc_algorithm_list(self) -> list[str]:
        return [item.strip() for item in self.oidc_algorithms.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
