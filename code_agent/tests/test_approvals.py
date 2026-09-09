from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from app.storage import Conflict, Storage, value_hash


def _active_run(
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
    assert storage.transition_to_running(run["run_id"])
    return run["run_id"]


def test_approval_replay_and_conflicting_decision(
    storage: Storage,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
) -> None:
    run_id = _active_run(
        storage, session_payload, run_payload, "decision"
    )
    arguments = {"command": "python -m pytest"}
    approval = storage.create_approval(
        run_id,
        "call-shell",
        arguments,
        "Runs code",
        (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )
    correct_hash = value_hash(arguments)

    with pytest.raises(Conflict, match="arguments changed"):
        storage.decide_approval(
            approval["approval_id"],
            "decision-key",
            {
                "decision": "allow_once",
                "expected_arguments_hash": "sha256:" + "0" * 64,
                "comment": None,
            },
        )

    payload = {
        "decision": "allow_once",
        "expected_arguments_hash": correct_hash,
        "comment": "Reviewed",
    }
    first, first_replay = storage.decide_approval(
        approval["approval_id"], "decision-key", payload
    )
    replay, second_replay = storage.decide_approval(
        approval["approval_id"], "decision-key", dict(payload)
    )
    replay_with_new_key, third_replay = storage.decide_approval(
        approval["approval_id"], "another-key", dict(payload)
    )

    assert first == replay == replay_with_new_key
    assert first["decision"] == "allow_once"
    assert first_replay is False
    assert second_replay is True
    assert third_replay is True

    with pytest.raises(Conflict, match="different decision"):
        storage.decide_approval(
            approval["approval_id"],
            "other-key",
            {**payload, "decision": "deny"},
        )


def test_revoked_approval_is_stale(
    storage: Storage,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
) -> None:
    run_id = _active_run(storage, session_payload, run_payload, "stale")
    arguments = {"command": "python script.py"}
    approval = storage.create_approval(
        run_id,
        "call-stale",
        arguments,
        "Runs code",
        (datetime.now(UTC) + timedelta(minutes=5)).isoformat(),
    )
    storage.request_cancel(run_id, "No longer needed")

    with pytest.raises(Conflict, match="stale or revoked"):
        storage.decide_approval(
            approval["approval_id"],
            "decision-stale",
            {
                "decision": "allow_once",
                "expected_arguments_hash": value_hash(arguments),
                "comment": None,
            },
        )
