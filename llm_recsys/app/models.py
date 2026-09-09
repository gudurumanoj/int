from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RequestContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    locale: str = "en-US"
    device: str | None = None
    query: str | None = Field(default=None, max_length=300)
    interests: list[str] = Field(default_factory=list, max_length=10)


class RecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: str = Field(min_length=1, max_length=100)
    user_id: str = Field(min_length=1, max_length=100)
    session_id: str = Field(min_length=1, max_length=100)
    placement: str = Field(min_length=1, max_length=80)
    context: RequestContext = Field(default_factory=RequestContext)
    num_results: int = Field(default=5, ge=1, le=20)
    explain: bool = True


class FeedbackRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str = Field(min_length=1, max_length=100)
    tracking_token: str = Field(min_length=20, max_length=500)
    item_id: str = Field(min_length=1, max_length=100)
    action: Literal["click", "dismiss", "bookmark", "dwell_60s"]
    position: int = Field(ge=1)
    session_id: str = Field(min_length=1, max_length=100)
    client_timestamp: datetime
    dwell_ms: int | None = Field(default=None, ge=0)


class CatalogItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    title: str
    description: str
    category: str
    tags: list[str]
    placements: list[str]
    locales: list[str]
    tenants: list[str]
    available: bool = True


class LLMRankedItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    item_id: str
    rank: int = Field(ge=1)
    reason: str = Field(min_length=1, max_length=240)

    @field_validator("reason")
    @classmethod
    def normalize_reason(cls, value: str) -> str:
        return " ".join(value.split())


class LLMRerankResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[LLMRankedItem] = Field(min_length=1)
