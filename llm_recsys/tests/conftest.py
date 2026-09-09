from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app


@pytest.fixture
def client_factory(tmp_path: Path):
    clients: list[TestClient] = []

    def create(reranker=None) -> tuple[TestClient, object]:
        settings = Settings(
            database_path=tmp_path / f"test-{len(clients)}.db",
            catalog_path=Path(__file__).parents[1] / "app" / "data" / "items.json",
            hmac_secret="test-secret-that-is-long-and-random-enough",
            recommendation_ttl_seconds=1_800,
            idempotency_ttl_seconds=86_400,
            llm_shortlist_size=4,
            openai_api_key="",
            openai_timeout_seconds=0.2,
        )
        app = create_app(settings=settings, reranker=reranker)
        client = TestClient(app)
        clients.append(client)
        return client, app

    yield create

    for client in clients:
        client.close()
