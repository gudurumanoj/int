from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str = Field(min_length=1)


class RoutingHints(BaseModel):
    task_type: Literal["auto", "chat", "reasoning", "code", "extraction"] = "auto"
    strategy: Literal["auto", "direct", "cascade", "fan_out"] = "auto"
    max_latency_ms: int | None = Field(default=None, gt=0)
    max_cost_usd: float | None = Field(default=None, gt=0)
    quality: Literal["low", "balanced", "high"] = "balanced"
    model_override: str | None = None


class ResponseRequest(BaseModel):
    messages: list[Message] = Field(min_length=1)
    routing: RoutingHints = Field(default_factory=RoutingHints)
    response_format: Literal["text", "json"] = "text"
    temperature: float | None = Field(default=None, ge=0, le=2)
    max_tokens: int | None = Field(default=None, gt=0)
    stream: bool = False


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0


class ResponseSnapshot(BaseModel):
    id: str
    status: Literal["in_progress", "completed", "failed", "cancelled"]
    content: str | None = None
    model: str | None = None
    routing_metadata: dict
    usage: Usage = Field(default_factory=Usage)
    error: str | None = None
    event_cursor: int


class ModelTarget(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    base_url: str
    api_key_env: str
    upstream_model: str
    capabilities: frozenset[str]
    quality_tier: Literal["low", "balanced", "high"]
    p99_latency_ms: float = Field(gt=0)
    input_price_per_1k_tokens: float = Field(ge=0)
    output_price_per_1k_tokens: float = Field(ge=0)
    health: Literal["healthy", "degraded", "unavailable"]
    quality_gate: Literal["json_valid"] | None = None


class RegistryFile(BaseModel):
    version: str
    models: list[ModelTarget] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_ids(self) -> RegistryFile:
        ids = [model.id for model in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("model IDs must be unique")
        return self


class ModelRegistry:
    def __init__(self, path: Path):
        data = json.loads(path.read_text(encoding="utf-8"))
        parsed = RegistryFile.model_validate(data)
        self.version = parsed.version
        self.models = tuple(parsed.models)
        self.by_id = {model.id: model for model in self.models}


class UpstreamResult(BaseModel):
    content: str
    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)


class AttemptOutcome(BaseModel):
    model_id: str
    content: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


class ExecutionPlan(BaseModel):
    strategy: Literal["direct", "cascade", "fan_out"]
    models: list[str]
    fallback_on: set[str] = Field(
        default_factory=lambda: {"timeout", "overload", "retryable_error"}
    )
    retries: int
    quality_gate_id: str | None
    registry_version: str
    estimated_max_cost_usd: float
    estimated_p99_latency_ms: float
    deadline_ms: int
