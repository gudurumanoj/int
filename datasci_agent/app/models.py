from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CreateSessionRequest(StrictModel):
    title: str = Field(min_length=1, max_length=200)


class RegisterDatasetRequest(StrictModel):
    filename: str = Field(min_length=1, max_length=255)
    content: str


class DatasetRef(StrictModel):
    dataset_id: str
    version: int = Field(ge=1)
    sha256: str = Field(min_length=64, max_length=71)


class RunConfig(StrictModel):
    target_column: str | None = None
    time_column: str | None = None
    group_column: str | None = None
    max_steps: int = Field(default=12, ge=1, le=50)
    max_cost_usd: float = Field(default=3.0, gt=0)
    max_wall_time_seconds: int = Field(default=900, ge=1)
    approval_policy: Literal["risk_based", "always", "never"] = "risk_based"
    output_format: Literal["markdown_report"] = "markdown_report"
    random_seed: int = 42
    holdout_fraction: float = Field(default=0.2, gt=0, lt=0.5)


class CreateRunRequest(StrictModel):
    question: str = Field(min_length=1, max_length=4000)
    datasets: list[DatasetRef] = Field(min_length=1)
    parent_run_id: str | None = None
    config: RunConfig = Field(default_factory=RunConfig)


class CancelRequest(StrictModel):
    reason: str = Field(min_length=1, max_length=500)


class ApprovalDecisionRequest(StrictModel):
    decision: Literal["approve", "reject", "request_changes"]
    expected_proposal_hash: str
    expected_plan_version: int | None = Field(default=None, ge=1)
    comment: str | None = Field(default=None, max_length=1000)


StepType = Literal[
    "eda",
    "cleaning",
    "feature_eng",
    "modeling",
    "evaluation",
    "reporting",
]


class AnalysisStep(StrictModel):
    id: str = Field(pattern=r"^[A-Za-z0-9_-]+$")
    description: str = Field(min_length=1)
    depends_on: list[str] = Field(default_factory=list)
    step_type: StepType
    input_artifact_ids: list[str] = Field(default_factory=list)
    expected_outputs: list[str] = Field(default_factory=list)
    acceptance_checks: list[str] = Field(default_factory=list)
    risk_tags: list[str] = Field(default_factory=list)


class AnalysisPlan(StrictModel):
    plan_id: str = Field(min_length=1)
    version: int = Field(ge=1)
    parent_version: int | None = Field(default=None, ge=1)
    based_on_event_seq: int = Field(ge=0)
    steps: list[AnalysisStep] = Field(min_length=1)

    @field_validator("steps")
    @classmethod
    def unique_step_ids(cls, steps: list[AnalysisStep]) -> list[AnalysisStep]:
        ids = [step.id for step in steps]
        if len(ids) != len(set(ids)):
            raise ValueError("step IDs must be unique")
        return steps

    @model_validator(mode="after")
    def valid_dag(self) -> "AnalysisPlan":
        known: set[str] = set()
        all_ids = {step.id for step in self.steps}
        for step in self.steps:
            unknown = set(step.depends_on) - all_ids
            if unknown:
                raise ValueError(f"{step.id} has unknown dependencies: {unknown}")
            if any(dep not in known for dep in step.depends_on):
                raise ValueError("steps must be topologically ordered")
            known.add(step.id)
        return self


class WorkerProposal(StrictModel):
    code: str = Field(min_length=1)
    risk_tags: list[str] = Field(default_factory=list)
    expected_artifacts: list[str] = Field(default_factory=list)
    notes: str = ""


class RunnerResult(StrictModel):
    status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    stdout: str = ""
    artifacts: dict[str, str] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    error: dict[str, Any] | None = None
