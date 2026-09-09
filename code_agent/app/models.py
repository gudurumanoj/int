from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SessionCreate(StrictModel):
    workspace_id: str = Field(min_length=1, max_length=200)


class RunLimits(StrictModel):
    max_steps: int = Field(default=20, ge=1, le=100)
    max_wall_time_seconds: int = Field(default=300, ge=1, le=3600)
    model: str | None = None
    approval_policy: Literal["risk_based"] = "risk_based"


class RunCreate(StrictModel):
    input: str = Field(min_length=1, max_length=100_000)
    base_workspace_revision: str = Field(default="initial", min_length=1)
    config: RunLimits = Field(default_factory=RunLimits)


class CancelRequest(StrictModel):
    reason: str = Field(default="User requested cancellation", max_length=1000)


class ApprovalDecisionRequest(StrictModel):
    decision: Literal["allow_once", "deny"]
    expected_arguments_hash: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    comment: str | None = Field(default=None, max_length=2000)


class ToolCall(BaseModel):
    id: str
    name: str
    arguments: dict[str, Any]


class ModelTurn(BaseModel):
    summary: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    finish_reason: str | None = None


class ToolResult(BaseModel):
    tool_call_id: str
    tool_name: str
    status: Literal["succeeded", "failed", "cancelled", "timed_out"]
    stdout: str = ""
    stderr: str = ""
    exit_code: int | None = None
    error_code: str | None = None
    retryable: bool = False
    side_effect_summary: dict[str, Any] = Field(default_factory=dict)
    started_at: str
    finished_at: str
