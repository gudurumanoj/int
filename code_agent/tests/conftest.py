from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from app.config import Settings
from app.storage import Storage


@pytest.fixture
def storage(tmp_path: Path) -> Iterator[Storage]:
    store = Storage(tmp_path / "agent.db")
    store.initialize()
    try:
        yield store
    finally:
        store.close()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        openai_base_url="https://example.invalid/v1",
        openai_api_key="fake-key",
        openai_model="fake-model",
        openai_timeout_seconds=1,
        openai_max_retries=0,
        openai_temperature=0,
        openai_max_tokens=128,
        db_path=tmp_path / "agent.db",
        workspace_root=tmp_path / "workspaces",
        docker_image="unused:test",
        approval_ttl_seconds=60,
        sse_heartbeat_seconds=0.05,
    )


@pytest.fixture
def session_payload() -> dict[str, str]:
    return {"workspace_id": "workspace-test"}


@pytest.fixture
def run_payload() -> dict[str, object]:
    return {
        "input": "Create a small example",
        "base_workspace_revision": "initial",
        "config": {
            "max_steps": 5,
            "max_wall_time_seconds": 30,
            "model": None,
            "approval_policy": "risk_based",
        },
    }
