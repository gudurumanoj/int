from __future__ import annotations

from copy import deepcopy


def recommendation_payload() -> dict:
    return {
        "request_id": "req_7f",
        "user_id": "u_12345",
        "session_id": "sess_456",
        "placement": "homepage",
        "context": {
            "locale": "en-US",
            "device": "desktop",
            "interests": ["transformers", "volatility"],
        },
        "num_results": 3,
        "explain": True,
    }


def create_recommendations(client, key: str = "req-key") -> dict:
    response = client.post(
        "/v1/recommendations",
        headers={"Idempotency-Key": key},
        json=recommendation_payload(),
    )
    assert response.status_code == 200
    return response.json()


def feedback_payload(recommendations: dict) -> dict:
    first = recommendations["recommendations"][0]
    return {
        "event_id": "evt_98",
        "tracking_token": first["tracking_token"],
        "item_id": first["item_id"],
        "action": "click",
        "position": first["rank"],
        "session_id": "sess_456",
        "client_timestamp": "2026-09-08T12:00:00Z",
        "dwell_ms": None,
    }


def test_recommendation_idempotency_replay_and_conflict(client_factory):
    client, _ = client_factory()
    payload = recommendation_payload()
    headers = {"Idempotency-Key": "same-key"}

    first = client.post("/v1/recommendations", headers=headers, json=payload)
    replay = client.post("/v1/recommendations", headers=headers, json=payload)

    assert first.status_code == 200
    assert replay.status_code == 200
    assert replay.json() == first.json()

    changed = deepcopy(payload)
    changed["num_results"] = 2
    conflict = client.post("/v1/recommendations", headers=headers, json=changed)
    assert conflict.status_code == 409


def test_deterministic_token_overlap_ranking(client_factory):
    client, _ = client_factory()
    payload = recommendation_payload()
    payload["context"]["interests"] = ["python", "time-series"]
    payload["explain"] = False

    response = client.post(
        "/v1/recommendations",
        headers={"Idempotency-Key": "deterministic-key"},
        json=payload,
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"][0]["item_id"] == "paper_timeseries_features"
    assert body["metadata"] == {"route": "deterministic", "degraded": False}
    assert "guide_sql_basics" not in {
        item["item_id"] for item in body["recommendations"]
    }
    assert "score" not in body["recommendations"][0]


class InvalidReranker:
    async def rerank(self, request, candidates, num_results):
        return {
            "items": [
                {
                    "item_id": "invented_item",
                    "rank": 1,
                    "reason": "An unsupported recommendation.",
                }
            ]
        }


def test_invalid_llm_output_falls_back_to_deterministic_ranking(client_factory):
    client, _ = client_factory(reranker=InvalidReranker())

    response = client.post(
        "/v1/recommendations",
        headers={"Idempotency-Key": "invalid-llm"},
        json=recommendation_payload(),
    )

    assert response.status_code == 200
    body = response.json()
    assert body["recommendations"][0]["item_id"] == "paper_regime_transformers"
    assert body["metadata"] == {
        "route": "deterministic_fallback",
        "degraded": True,
    }


def test_tracking_token_tampering_is_rejected(client_factory):
    client, app = client_factory()
    recommendations = create_recommendations(client)
    payload = feedback_payload(recommendations)
    token = payload["tracking_token"]
    payload["tracking_token"] = ("A" if token[0] != "A" else "B") + token[1:]

    response = client.post(
        f"/v1/recommendations/{recommendations['recommendation_set_id']}/feedback",
        headers={"Idempotency-Key": "tampered-feedback"},
        json=payload,
    )

    assert response.status_code == 400
    assert app.state.database.feedback_count("demo") == 0


def test_feedback_duplicate_and_conflicts(client_factory):
    client, app = client_factory()
    recommendations = create_recommendations(client)
    set_id = recommendations["recommendation_set_id"]
    path = f"/v1/recommendations/{set_id}/feedback"
    payload = feedback_payload(recommendations)

    first = client.post(
        path, headers={"Idempotency-Key": "feedback-key"}, json=payload
    )
    replay = client.post(
        path, headers={"Idempotency-Key": "feedback-key"}, json=payload
    )
    duplicate = client.post(
        path, headers={"Idempotency-Key": "another-feedback-key"}, json=payload
    )

    assert first.status_code == 201
    assert first.json()["deduplicated"] is False
    assert replay.status_code == 201
    assert replay.json() == first.json()
    assert duplicate.status_code == 200
    assert duplicate.json()["deduplicated"] is True
    assert app.state.database.feedback_count("demo") == 1

    changed = deepcopy(payload)
    changed["action"] = "bookmark"
    event_conflict = client.post(
        path, headers={"Idempotency-Key": "changed-event-key"}, json=changed
    )
    assert event_conflict.status_code == 409

    changed_key_payload = deepcopy(payload)
    changed_key_payload["dwell_ms"] = 10
    key_conflict = client.post(
        path,
        headers={"Idempotency-Key": "feedback-key"},
        json=changed_key_payload,
    )
    assert key_conflict.status_code == 409
    assert app.state.database.feedback_count("demo") == 1
