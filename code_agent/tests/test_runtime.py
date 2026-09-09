from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from app.config import Settings
from app.models import ModelTurn
from app.runtime import AgentRuntime
from app.storage import Conflict, Storage
from app.tools import ToolRegistry


class FakeChatClient:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> ModelTurn:
        self.calls.append(
            {"messages": messages, "tools": tools, "model": model}
        )
        return ModelTurn(
            summary="Created the requested example.",
            tool_calls=[],
            finish_reason="stop",
        )


class FakeRunner:
    def __init__(self) -> None:
        self.cancelled: list[str] = []

    async def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)

    async def run(
        self,
        run_id: str,
        workspace: Path,
        command: str,
        timeout_seconds: float,
    ) -> tuple[str, str, int | None, str]:
        raise AssertionError("Tests must not execute Docker")


def _runtime(
    storage: Storage, settings: Settings
) -> tuple[AgentRuntime, FakeChatClient, FakeRunner]:
    client = FakeChatClient()
    runner = FakeRunner()
    tools = ToolRegistry(settings.workspace_root, runner)
    runtime = AgentRuntime(
        storage,
        client,
        tools,
        runner,
        settings,
    )
    return runtime, client, runner


def _queued_run(
    storage: Storage,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
    suffix: str,
) -> str:
    session, _, _ = storage.create_session(
        f"session-{suffix}", session_payload
    )
    run, _, _ = storage.create_run(
        session["session_id"], f"run-{suffix}", run_payload
    )
    return run["run_id"]


@pytest.mark.asyncio
async def test_runtime_uses_injected_fake_client(
    storage: Storage,
    settings: Settings,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
) -> None:
    runtime, client, _ = _runtime(storage, settings)
    run_id = _queued_run(
        storage, session_payload, run_payload, "complete"
    )

    await runtime.execute_run(run_id)

    assert storage.get_run(run_id)["status"] == "completed"
    assert storage.get_run(run_id)["result"] == {
        "answer": "Created the requested example."
    }
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_cancellation_is_idempotent_and_stops_active_runner(
    storage: Storage,
    settings: Settings,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
) -> None:
    runtime, _, runner = _runtime(storage, settings)
    run_id = _queued_run(
        storage, session_payload, run_payload, "cancel"
    )
    payload = {"reason": "Stop now"}

    first, first_replay = await runtime.cancel_run(
        run_id, "cancel-key", payload
    )
    replay, second_replay = await runtime.cancel_run(
        run_id, "cancel-key", dict(payload)
    )

    assert first == replay == {"run_id": run_id, "status": "cancelled"}
    assert first_replay is False
    assert second_replay is True
    assert runner.cancelled == [run_id]

    with pytest.raises(Conflict, match="another payload"):
        await runtime.cancel_run(
            run_id, "cancel-key", {"reason": "Different reason"}
        )

    terminal, terminal_replay = await runtime.cancel_run(
        run_id, "a-new-key", {"reason": "Still stop"}
    )
    assert terminal == {"run_id": run_id, "status": "cancelled"}
    assert terminal_replay is False
    assert runner.cancelled == [run_id]
