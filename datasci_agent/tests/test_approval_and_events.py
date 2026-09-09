from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app

from .conftest import (
    FakeLLM,
    FakeRunner,
    create_run,
    create_session,
    make_settings,
    register_dataset,
)


def test_approval_rejects_stale_hash_then_resumes_exact_proposal(
    tmp_path: Path,
) -> None:
    fake_llm = FakeLLM(risk_tags=["row_drop"])
    fake_runner = FakeRunner()
    app = create_app(
        make_settings(tmp_path / "approval-data"),
        llm=fake_llm,
        runner=fake_runner,
        auto_execute=True,
    )
    with TestClient(app) as client:
        session_id = create_session(client)
        dataset = register_dataset(client)
        run, _ = create_run(client, session_id, dataset)
        run_id = run["run_id"]

        snapshot = client.get(f"/v1/datasci/runs/{run_id}").json()
        assert snapshot["status"] == "waiting_for_approval"
        approval = app.state.db.find_step_approval(run_id, 1, "step_01")
        assert approval is not None
        url = f"/v1/datasci/approval-requests/{approval['id']}/decision"

        stale = client.put(
            url,
            headers={"Idempotency-Key": "stale-decision"},
            json={
                "decision": "approve",
                "expected_proposal_hash": "sha256:stale",
                "expected_plan_version": 1,
            },
        )
        assert stale.status_code == 409
        assert fake_runner.calls == []

        approved = client.put(
            url,
            headers={"Idempotency-Key": "exact-decision"},
            json={
                "decision": "approve",
                "expected_proposal_hash": approval["proposal_hash"],
                "expected_plan_version": approval["plan_version"],
            },
        )
        assert approved.status_code == 200
        assert approved.json()["proposal_hash"] == approval["proposal_hash"]
        assert client.get(f"/v1/datasci/runs/{run_id}").json()["status"] == "succeeded"
        assert fake_runner.calls == [(run_id, 1, "step_01")]
        assert fake_llm.planner_calls == fake_llm.worker_calls == 1


def test_sse_replays_only_events_after_last_event_id(client: TestClient) -> None:
    session_id = create_session(client)
    dataset = register_dataset(client)
    run, _ = create_run(client, session_id, dataset)
    run_id = run["run_id"]
    client.app.state.db.append_event(run_id, "test.marker", {"value": 7})
    cancelled = client.post(
        f"/v1/datasci/runs/{run_id}/cancel",
        headers={"Idempotency-Key": "cancel-for-stream"},
        json={"reason": "finish stream"},
    )
    assert cancelled.status_code == 200

    response = client.get(
        f"/v1/datasci/runs/{run_id}/events",
        headers={"Last-Event-ID": "1", "Accept": "text/event-stream"},
    )
    assert response.status_code == 200
    event_ids = [
        int(line.removeprefix("id: "))
        for line in response.text.splitlines()
        if line.startswith("id: ")
    ]
    assert event_ids == [2, 3, 4]
    assert "event: test.marker" in response.text
    assert "event: run.queued" not in response.text
