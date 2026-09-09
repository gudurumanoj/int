from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from app.domain import ModelTarget, ResponseRequest, UpstreamResult
from app.main import create_app
from app.upstream import RetryableUpstreamError


ROOT = Path(__file__).resolve().parents[1]
Handler = Callable[[ModelTarget, ResponseRequest], Awaitable[UpstreamResult]]


class FakeUpstream:
    def __init__(self, handlers: dict[str, Handler] | None = None):
        self.handlers = handlers or {}
        self.calls: list[str] = []
        self.closed = False

    async def generate(
        self,
        target: ModelTarget,
        request: ResponseRequest,
        timeout_seconds: float,
    ) -> UpstreamResult:
        self.calls.append(target.id)
        handler = self.handlers.get(target.id)
        if handler:
            return await handler(target, request)
        content = '{"model":"' + target.id + '"}' if request.response_format == "json" else "ok"
        return UpstreamResult(content=content, input_tokens=10, output_tokens=5)

    async def close(self) -> None:
        self.closed = True


@asynccontextmanager
async def client_for(tmp_path: Path, upstream: FakeUpstream, **overrides):
    values = {
        "sqlite_path": tmp_path / "test.db",
        "model_registry_path": ROOT / "models.json",
        "default_timeout_ms": 5_000,
        "default_retries": 0,
        **overrides,
    }
    app = create_app(Settings(**values), upstream)
    transport = httpx.ASGITransport(app=app)
    try:
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://test",
        ) as client:
            yield client, app
    finally:
        await app.state.orchestrator.shutdown()


async def wait_terminal(client: httpx.AsyncClient, response_id: str) -> dict:
    for _ in range(100):
        response = await client.get(f"/v1/responses/{response_id}")
        body = response.json()
        if body["status"] != "in_progress":
            return body
        await asyncio.sleep(0.01)
    pytest.fail("response did not reach a terminal state")


def request_body(**routing) -> dict:
    return {
        "messages": [{"role": "user", "content": "hello"}],
        "routing": routing,
    }


@pytest.mark.asyncio
async def test_idempotency_replay_and_conflict(tmp_path: Path) -> None:
    fake = FakeUpstream()
    async with client_for(tmp_path, fake) as (client, _):
        headers = {"Idempotency-Key": "same-key"}
        first = await client.post(
            "/v1/responses",
            headers=headers,
            json=request_body(quality="balanced"),
        )
        second = await client.post(
            "/v1/responses",
            headers=headers,
            json=request_body(quality="balanced"),
        )
        conflict = await client.post(
            "/v1/responses",
            headers=headers,
            json=request_body(quality="high"),
        )

        assert first.status_code == 202
        assert second.status_code == 200
        assert second.headers["Idempotent-Replay"] == "true"
        assert first.json()["id"] == second.json()["id"]
        assert conflict.status_code == 409
        await wait_terminal(client, first.json()["id"])
        assert len(fake.calls) == 1


@pytest.mark.asyncio
async def test_direct_and_cascade_model_selection(tmp_path: Path) -> None:
    async def invalid(_: ModelTarget, __: ResponseRequest) -> UpstreamResult:
        return UpstreamResult(content="not json", input_tokens=3, output_tokens=2)

    fake = FakeUpstream({"fast": invalid})
    async with client_for(tmp_path, fake) as (client, app):
        direct = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "direct"},
            json=request_body(quality="high", strategy="direct"),
        )
        direct_body = await wait_terminal(client, direct.json()["id"])
        assert direct_body["status"] == "completed"
        assert direct_body["model"] == "quality"
        assert direct_body["routing_metadata"]["strategy"] == "direct"

        cascade_payload = request_body(
            task_type="extraction",
            quality="balanced",
            strategy="cascade",
        )
        cascade_payload["response_format"] = "json"
        cascade = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "cascade"},
            json=cascade_payload,
        )
        cascade_body = await wait_terminal(client, cascade.json()["id"])
        assert cascade_body["status"] == "completed"
        assert cascade_body["model"] == "balanced"
        assert cascade_body["routing_metadata"]["strategy"] == "cascade"
        assert fake.calls == ["quality", "fast", "balanced"]

        reservation = app.state.orchestrator.database.connection.execute(
            "SELECT status, actual_usd FROM cost_reservations WHERE response_id = ?",
            (direct_body["id"],),
        ).fetchone()
        assert reservation["status"] == "settled"
        assert reservation["actual_usd"] > 0


@pytest.mark.asyncio
async def test_failover_is_retryable_error_policy_not_strategy(
    tmp_path: Path,
) -> None:
    async def overloaded(_: ModelTarget, __: ResponseRequest) -> UpstreamResult:
        raise RetryableUpstreamError("overloaded")

    fake = FakeUpstream({"quality": overloaded})
    async with client_for(tmp_path, fake) as (client, _):
        created = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "failover"},
            json=request_body(quality="high", strategy="direct"),
        )
        body = await wait_terminal(client, created.json()["id"])

        assert body["status"] == "completed"
        assert body["model"] == "balanced"
        assert body["routing_metadata"]["strategy"] == "direct"
        assert fake.calls == ["quality", "balanced"]


@pytest.mark.asyncio
async def test_budget_and_deadline_rejection_before_dispatch(tmp_path: Path) -> None:
    fake = FakeUpstream()
    async with client_for(tmp_path, fake) as (client, _):
        budget = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "budget"},
            json=request_body(max_cost_usd=0.000000001),
        )
        deadline = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "deadline"},
            json=request_body(max_latency_ms=100),
        )
        assert budget.status_code == 422
        assert "worst-case cost" in budget.json()["detail"]
        assert deadline.status_code == 422
        assert "p99 latency" in deadline.json()["detail"]
        assert fake.calls == []


@pytest.mark.asyncio
async def test_delete_propagates_cancellation_and_is_idempotent(tmp_path: Path) -> None:
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def block(_: ModelTarget, __: ResponseRequest) -> UpstreamResult:
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("unreachable")

    fake = FakeUpstream({"balanced": block})
    async with client_for(tmp_path, fake) as (client, app):
        created = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "cancel"},
            json=request_body(quality="balanced"),
        )
        response_id = created.json()["id"]
        await asyncio.wait_for(started.wait(), timeout=1)

        first = await client.delete(f"/v1/responses/{response_id}")
        second = await client.delete(f"/v1/responses/{response_id}")
        await asyncio.wait_for(cancelled.wait(), timeout=1)

        assert first.json()["status"] == "cancelled"
        assert second.json()["status"] == "cancelled"
        reservation = app.state.orchestrator.database.connection.execute(
            "SELECT status FROM cost_reservations WHERE response_id = ?",
            (response_id,),
        ).fetchone()
        assert reservation["status"] == "settled"


@pytest.mark.asyncio
async def test_fan_out_selects_valid_result_and_cancels_slow_branch(
    tmp_path: Path,
) -> None:
    slow_started = asyncio.Event()
    slow_cancelled = asyncio.Event()

    async def fast_valid(_: ModelTarget, __: ResponseRequest) -> UpstreamResult:
        await asyncio.sleep(0.02)
        return UpstreamResult(content='{"winner":true}', input_tokens=2, output_tokens=2)

    async def slow(_: ModelTarget, __: ResponseRequest) -> UpstreamResult:
        slow_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            slow_cancelled.set()
            raise
        raise AssertionError("unreachable")

    fake = FakeUpstream(
        {
            "fast": fast_valid,
            "balanced": slow,
            "quality": slow,
        }
    )
    async with client_for(tmp_path, fake) as (client, _):
        payload = request_body(
            task_type="extraction",
            quality="high",
            strategy="fan_out",
        )
        payload["response_format"] = "json"
        created = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "fan"},
            json=payload,
        )
        await asyncio.wait_for(slow_started.wait(), timeout=1)
        body = await wait_terminal(client, created.json()["id"])
        await asyncio.wait_for(slow_cancelled.wait(), timeout=1)

        assert body["status"] == "completed"
        assert body["model"] == "fast"
        assert body["content"] == '{"winner":true}'
        assert body["routing_metadata"]["strategy"] == "fan_out"


@pytest.mark.asyncio
async def test_sse_replays_only_events_after_last_event_id(tmp_path: Path) -> None:
    fake = FakeUpstream()
    async with client_for(tmp_path, fake) as (client, _):
        created = await client.post(
            "/v1/responses",
            headers={"Idempotency-Key": "events"},
            json=request_body(),
        )
        response_id = created.json()["id"]
        await wait_terminal(client, response_id)

        replay = await client.get(
            f"/v1/responses/{response_id}/events",
            headers={"Last-Event-ID": "1", "Accept": "text/event-stream"},
        )
        assert replay.status_code == 200
        assert "event: response.created" not in replay.text
        assert "id: 2\nevent: response.output_delta" in replay.text
        assert "id: 3\nevent: response.completed" in replay.text
