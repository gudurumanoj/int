from __future__ import annotations

from fastapi.testclient import TestClient

from .conftest import create_run, create_session, register_dataset


def test_idempotency_replays_and_conflicts(client: TestClient) -> None:
    first = client.post(
        "/v1/datasci/sessions",
        headers={"Idempotency-Key": "same-key"},
        json={"title": "Original"},
    )
    replay = client.post(
        "/v1/datasci/sessions",
        headers={"Idempotency-Key": "same-key"},
        json={"title": "Original"},
    )
    conflict = client.post(
        "/v1/datasci/sessions",
        headers={"Idempotency-Key": "same-key"},
        json={"title": "Different"},
    )

    assert first.status_code == replay.status_code == 201
    assert first.json() == replay.json()
    assert conflict.status_code == 409

    dataset = register_dataset(client)
    run_first, payload = create_run(client, first.json()["session_id"], dataset)
    run_replay = client.post(
        f"/v1/datasci/sessions/{first.json()['session_id']}/runs",
        headers={"Idempotency-Key": "run-1"},
        json=payload,
    )
    changed = {**payload, "question": "A different question"}
    run_conflict = client.post(
        f"/v1/datasci/sessions/{first.json()['session_id']}/runs",
        headers={"Idempotency-Key": "run-1"},
        json=changed,
    )
    assert run_replay.json() == run_first
    assert run_conflict.status_code == 409


def test_cancellation_is_idempotent(client: TestClient, fake_runner) -> None:
    session_id = create_session(client)
    dataset = register_dataset(client)
    run, _ = create_run(client, session_id, dataset)
    url = f"/v1/datasci/runs/{run['run_id']}/cancel"
    body = {"reason": "user stopped it"}

    first = client.post(
        url, headers={"Idempotency-Key": "cancel-1"}, json=body
    )
    replay = client.post(
        url, headers={"Idempotency-Key": "cancel-1"}, json=body
    )
    conflict = client.post(
        url,
        headers={"Idempotency-Key": "cancel-1"},
        json={"reason": "different reason"},
    )

    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert first.json()["status"] == "cancelled"
    assert conflict.status_code == 409
    assert fake_runner.cancelled == [run["run_id"]]
