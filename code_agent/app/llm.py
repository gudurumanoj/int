from __future__ import annotations

import json
from typing import Any, Protocol

from openai import AsyncOpenAI

from app.config import Settings
from app.models import ModelTurn, ToolCall


class ChatClient(Protocol):
    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> ModelTurn: ...


class OpenAIChatClient:
    """Thin adapter around an OpenAI-compatible Chat Completions endpoint."""

    def __init__(self, settings: Settings):
        if not settings.openai_api_key:
            raise ValueError("OPENAI_API_KEY must be configured")
        self.settings = settings
        self.client = AsyncOpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            timeout=settings.openai_timeout_seconds,
            max_retries=settings.openai_max_retries,
        )

    async def complete(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        *,
        model: str | None = None,
    ) -> ModelTurn:
        response = await self.client.chat.completions.create(
            model=model or self.settings.openai_model,
            messages=messages,  # type: ignore[arg-type]
            tools=tools,  # type: ignore[arg-type]
            tool_choice="auto",
            temperature=self.settings.openai_temperature,
            max_tokens=self.settings.openai_max_tokens,
        )
        choice = response.choices[0]
        message = choice.message
        calls = [
            ToolCall(
                id=call.id,
                name=call.function.name,
                arguments=json.loads(call.function.arguments),
            )
            for call in (message.tool_calls or [])
        ]
        return ModelTurn(
            summary=message.content or "",
            tool_calls=calls,
            finish_reason=choice.finish_reason,
        )
