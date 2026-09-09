from __future__ import annotations

import pytest

from app.storage import Conflict, Storage


def test_session_idempotency_replays_and_conflicts(
    storage: Storage, session_payload: dict[str, str]
) -> None:
    first, first_status, first_replay = storage.create_session(
        "session-key", session_payload
    )
    second, second_status, second_replay = storage.create_session(
        "session-key", dict(session_payload)
    )

    assert second == first
    assert (first_status, second_status) == (201, 201)
    assert first_replay is False
    assert second_replay is True

    with pytest.raises(Conflict, match="another payload"):
        storage.create_session(
            "session-key", {"workspace_id": "different-workspace"}
        )


def test_run_idempotency_does_not_create_duplicate_run(
    storage: Storage,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
) -> None:
    session, _, _ = storage.create_session(
        "session-key", session_payload
    )
    first, _, first_replay = storage.create_run(
        session["session_id"], "run-key", run_payload
    )
    second, _, second_replay = storage.create_run(
        session["session_id"], "run-key", run_payload
    )

    assert first["run_id"] == second["run_id"]
    assert first_replay is False
    assert second_replay is True

    changed = {**run_payload, "input": "A different task"}
    with pytest.raises(Conflict, match="another payload"):
        storage.create_run(
            session["session_id"], "run-key", changed
        )
