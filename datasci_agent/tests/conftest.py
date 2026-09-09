from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from app.models import AnalysisPlan, RunnerResult, WorkerProposal


class FakeLLM:
    model = "fake-model"

    def __init__(self, risk_tags: list[str] | None = None):
        self.risk_tags = risk_tags or []
        self.planner_calls = 0
        self.worker_calls = 0

    async def plan(
        self,
        question: str,
        datasets: list[dict[str, Any]],
        profiles: list[dict[str, Any]],
        config: dict[str, Any],
        based_on_event_seq: int,
    ) -> AnalysisPlan:
        self.planner_calls += 1
        return AnalysisPlan.model_validate(
            {
                "plan_id": "plan_test",
                "version": 1,
                "parent_version": None,
                "based_on_event_seq": based_on_event_seq,
                "steps": [
                    {
                        "id": "step_01",
                        "description": "Write deterministic summary",
                        "depends_on": [],
                        "step_type": "eda",
                        "input_artifact_ids": [],
                        "expected_outputs": ["summary.json"],
                        "acceptance_checks": ["summary exists"],
                        "risk_tags": self.risk_tags,
                    }
                ],
            }
        )

    async def propose_step(self, *args: Any, **kwargs: Any) -> WorkerProposal:
        self.worker_calls += 1
        return WorkerProposal(
            code=(
                "from pathlib import Path\n"
                "Path('/output/summary.json').write_text('{}')\n"
            ),
            risk_tags=self.risk_tags,
            expected_artifacts=["summary.json"],
        )


class FakeRunner:
    image = "fake-sandbox@sha256:test"

    def __init__(self):
        self.calls: list[tuple[str, int, str]] = []
        self.cancelled: list[str] = []

    async def run(
        self,
        run_id: str,
        plan_version: int,
        step_id: str,
        code: str,
        dataset_path: str,
    ) -> RunnerResult:
        self.calls.append((run_id, plan_version, step_id))
        return RunnerResult(
            status="succeeded",
            stdout="fake execution",
            artifacts={"summary.json": '{"rows": 2}'},
            metrics={"rows": 2.0},
        )

    async def cancel(self, run_id: str) -> None:
        self.cancelled.append(run_id)


def make_settings(data_dir: Path) -> Settings:
    return Settings(
        data_dir=data_dir,
        openai_base_url="https://invalid.test/v1",
        openai_api_key="fake",
        openai_model="fake-model",
        openai_timeout_seconds=1,
        openai_max_retries=0,
        openai_temperature=0,
        openai_max_tokens=500,
        docker_image="never-run",
        docker_cpus="1",
        docker_memory="256m",
        docker_timeout_seconds=1,
    )


@pytest.fixture
def fake_llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def fake_runner() -> FakeRunner:
    return FakeRunner()


@pytest.fixture
def client(tmp_path: Path, fake_llm: FakeLLM, fake_runner: FakeRunner):
    app = create_app(
        make_settings(tmp_path / "data"),
        llm=fake_llm,
        runner=fake_runner,
        auto_execute=False,
    )
    with TestClient(app) as test_client:
        yield test_client


def create_session(client: TestClient, key: str = "session-1") -> str:
    response = client.post(
        "/v1/datasci/sessions",
        headers={"Idempotency-Key": key},
        json={"title": "Test analysis"},
    )
    assert response.status_code == 201
    return response.json()["session_id"]


def register_dataset(client: TestClient, content: str = "x,target\n1,0\n2,1\n"):
    response = client.post(
        "/v1/datasci/datasets",
        json={"filename": "sample.csv", "content": content},
    )
    assert response.status_code == 201
    return response.json()


def create_run(client: TestClient, session_id: str, dataset: dict[str, Any]):
    payload = {
        "question": "What predicts target?",
        "datasets": [
            {
                "dataset_id": dataset["dataset_id"],
                "version": dataset["version"],
                "sha256": dataset["sha256"],
            }
        ],
        "config": {"target_column": "target"},
    }
    response = client.post(
        f"/v1/datasci/sessions/{session_id}/runs",
        headers={"Idempotency-Key": "run-1"},
        json=payload,
    )
    assert response.status_code == 202
    return response.json(), payload
