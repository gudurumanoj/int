from __future__ import annotations

import hashlib
from pathlib import Path

from fastapi.testclient import TestClient


def test_dataset_registration_is_hash_verified_and_immutable(
    client: TestClient,
) -> None:
    original = "feature,target\n1,0\n2,1\n"
    changed = "feature,target\n1,0\n2,1\n3,0\n"

    first = client.post(
        "/v1/datasci/datasets",
        json={"filename": "signals.csv", "content": original},
    )
    duplicate = client.post(
        "/v1/datasci/datasets",
        json={"filename": "signals.csv", "content": original},
    )
    second = client.post(
        "/v1/datasci/datasets",
        json={"filename": "signals.csv", "content": changed},
    )

    assert first.status_code == duplicate.status_code == second.status_code == 201
    assert first.json() == duplicate.json()
    assert first.json()["sha256"] == hashlib.sha256(original.encode()).hexdigest()
    assert second.json()["dataset_id"] == first.json()["dataset_id"]
    assert second.json()["version"] == 2
    assert second.json()["sha256"] != first.json()["sha256"]

    db = client.app.state.db
    old_record = db.get_dataset(
        first.json()["dataset_id"], first.json()["version"], first.json()["sha256"]
    )
    new_record = db.get_dataset(
        second.json()["dataset_id"], second.json()["version"], second.json()["sha256"]
    )
    assert Path(old_record["path"]).read_text(encoding="utf-8") == original
    assert Path(new_record["path"]).read_text(encoding="utf-8") == changed
