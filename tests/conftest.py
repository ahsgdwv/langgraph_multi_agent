from __future__ import annotations

import os
from pathlib import Path

import pytest

os.environ.setdefault("AUTH_MODE", "dev")
os.environ.setdefault("DATABASE_URL", "sqlite:///data/test_app.db")
os.environ.setdefault("LOG_JSON", "false")


@pytest.fixture(autouse=True)
def clean_database():
    from infra import models  # noqa: F401
    from infra.db import Base, get_engine

    engine = get_engine()
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    yield


@pytest.fixture(scope="session", autouse=True)
def ensure_test_dirs():
    Path("data").mkdir(exist_ok=True)
