from __future__ import annotations

from app.storage import Storage


def test_event_replay_is_strictly_after_sequence(
    storage: Storage,
    session_payload: dict[str, str],
    run_payload: dict[str, object],
) -> None:
    session, _, _ = storage.create_session(
        "session-key", session_payload
    )
    run, _, _ = storage.create_run(
        session["session_id"], "run-key", run_payload
    )
    run_id = run["run_id"]
    # Sequence 1 is run.queued.
    assert storage.append_event(
        run_id, "assistant.progress", {"summary": "one"}
    ) == 2
    assert storage.append_event(
        run_id, "assistant.progress", {"summary": "two"}
    ) == 3

    replay = storage.list_events_after(run_id, 1)

    assert [event["sequence"] for event in replay] == [2, 3]
    assert [event["data"]["summary"] for event in replay] == ["one", "two"]
    assert storage.list_events_after(run_id, 3) == []
