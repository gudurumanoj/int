from __future__ import annotations

import asyncio
import os
from typing import Protocol

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    RateLimitError,
)

from app.config import Settings
from app.domain import ModelTarget, ResponseRequest, UpstreamResult


class RetryableUpstreamError(Exception):
    pass


class FatalUpstreamError(Exception):
    pass


class Upstream(Protocol):
    async def generate(
        self,
        target: ModelTarget,
        request: ResponseRequest,
        timeout_seconds: float,
    ) -> UpstreamResult: ...

    async def close(self) -> None: ...


class OpenAIUpstream:
    """OpenAI-compatible adapter; tests replace this with an in-memory fake."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.clients: dict[str, AsyncOpenAI] = {}

    def _client(self, target: ModelTarget) -> AsyncOpenAI:
        if target.id in self.clients:
            return self.clients[target.id]
        key_env = target.api_key_env or self.settings.default_api_key_env
        api_key = os.getenv(key_env)
        if not api_key:
            raise FatalUpstreamError(f"missing API key environment variable: {key_env}")
        client = AsyncOpenAI(api_key=api_key, base_url=target.base_url)
        self.clients[target.id] = client
        return client

    async def generate(
        self,
        target: ModelTarget,
        request: ResponseRequest,
        timeout_seconds: float,
    ) -> UpstreamResult:
        try:
            async with asyncio.timeout(timeout_seconds):
                result = await self._client(target).chat.completions.create(
                    model=target.upstream_model,
                    messages=[message.model_dump() for message in request.messages],
                    temperature=request.temperature,
                    max_tokens=request.max_tokens,
                    response_format={"type": "json_object"}
                    if request.response_format == "json"
                    else None,
                    stream=False,
                )
        except (APITimeoutError, APIConnectionError, RateLimitError, TimeoutError) as exc:
            raise RetryableUpstreamError(type(exc).__name__) from exc
        except APIStatusError as exc:
            if exc.status_code >= 500:
                raise RetryableUpstreamError(f"upstream HTTP {exc.status_code}") from exc
            raise FatalUpstreamError(f"upstream HTTP {exc.status_code}") from exc

        content = result.choices[0].message.content
        if not content:
            raise FatalUpstreamError("upstream returned empty content")
        usage = result.usage
        return UpstreamResult(
            content=content,
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
        )

    async def close(self) -> None:
        await asyncio.gather(
            *(client.close() for client in self.clients.values()),
            return_exceptions=True,
        )
