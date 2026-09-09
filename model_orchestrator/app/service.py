from __future__ import annotations

import asyncio
import hashlib
import json
import time
from contextlib import suppress
from dataclasses import dataclass, field

from app.config import Settings
from app.database import Database
from app.domain import (
    AttemptOutcome,
    ExecutionPlan,
    ModelRegistry,
    ModelTarget,
    ResponseRequest,
    ResponseSnapshot,
)
from app.routing import Router
from app.upstream import (
    FatalUpstreamError,
    RetryableUpstreamError,
    Upstream,
)


class ExecutionFailed(Exception):
    pass


@dataclass
class ExecutionContext:
    cancellation: asyncio.Event = field(default_factory=asyncio.Event)
    tasks: set[asyncio.Task] = field(default_factory=set)
    attempt_number: int = 0

    def next_attempt(self) -> int:
        self.attempt_number += 1
        return self.attempt_number


class Orchestrator:
    def __init__(
        self,
        settings: Settings,
        registry: ModelRegistry,
        database: Database,
        upstream: Upstream,
    ):
        self.settings = settings
        self.registry = registry
        self.database = database
        self.upstream = upstream
        self.router = Router(registry, settings)
        self.active: dict[str, ExecutionContext] = {}

    def resolve_defaults(self, request: ResponseRequest) -> ResponseRequest:
        return request.model_copy(
            update={
                "temperature": request.temperature
                if request.temperature is not None
                else self.settings.default_temperature,
                "max_tokens": request.max_tokens
                if request.max_tokens is not None
                else self.settings.default_max_tokens,
            }
        )

    def create(
        self,
        tenant_id: str,
        idempotency_key: str,
        request: ResponseRequest,
    ) -> tuple[ResponseSnapshot, bool]:
        request = self.resolve_defaults(request)
        plan = self.router.plan(request)
        canonical = json.dumps(
            request.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        )
        payload_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
        snapshot, replayed = self.database.create_or_replay(
            tenant_id=tenant_id,
            idempotency_key=idempotency_key,
            payload_hash=payload_hash,
            request_json=canonical,
            plan=plan,
        )
        if not replayed:
            context = ExecutionContext()
            self.active[snapshot.id] = context
            task = asyncio.create_task(
                self._run(snapshot.id, request, plan, context),
                name=f"orchestrate-{snapshot.id}",
            )
            context.tasks.add(task)
        return snapshot, replayed

    async def cancel(self, response_id: str) -> ResponseSnapshot | None:
        snapshot = self.database.get(response_id)
        if not snapshot:
            return None
        context = self.active.get(response_id)
        if context:
            context.cancellation.set()
            current = asyncio.current_task()
            for task in list(context.tasks):
                if task is not current and not task.done():
                    task.cancel()
        self.database.cancel(response_id)
        return self.database.get(response_id)

    async def _run(
        self,
        response_id: str,
        request: ResponseRequest,
        plan: ExecutionPlan,
        context: ExecutionContext,
    ) -> None:
        deadline_at = asyncio.get_running_loop().time() + plan.deadline_ms / 1000
        try:
            if plan.strategy == "cascade":
                outcome = await self._cascade(
                    response_id, request, plan, context, deadline_at
                )
            elif plan.strategy == "fan_out":
                outcome = await self._fan_out(
                    response_id, request, plan, context, deadline_at
                )
            else:
                outcome = await self._direct(
                    response_id, request, plan, context, deadline_at
                )
            if context.cancellation.is_set():
                self.database.cancel(response_id)
            else:
                self.database.complete(
                    response_id,
                    outcome.content,
                    outcome.model_id,
                )
        except asyncio.CancelledError:
            self.database.cancel(response_id)
        except Exception as exc:
            self.database.fail(response_id, str(exc))
        finally:
            context.tasks.discard(asyncio.current_task())
            if self.active.get(response_id) is context:
                self.active.pop(response_id, None)

    async def _direct(
        self,
        response_id: str,
        request: ResponseRequest,
        plan: ExecutionPlan,
        context: ExecutionContext,
        deadline_at: float,
    ) -> AttemptOutcome:
        last_retryable: Exception | None = None
        for model_id in plan.models:
            target = self.registry.by_id[model_id]
            try:
                outcome = await self._call_with_retries(
                    response_id,
                    target,
                    request,
                    plan.retries,
                    context,
                    deadline_at,
                )
            except RetryableUpstreamError as exc:
                last_retryable = exc
                continue
            if plan.quality_gate_id and not self._valid_json(outcome.content):
                raise ExecutionFailed("selected model returned invalid JSON")
            return outcome
        raise ExecutionFailed(f"all failover targets failed: {last_retryable}")

    async def _cascade(
        self,
        response_id: str,
        request: ResponseRequest,
        plan: ExecutionPlan,
        context: ExecutionContext,
        deadline_at: float,
    ) -> AttemptOutcome:
        for model_id in plan.models:
            target = self.registry.by_id[model_id]
            try:
                outcome = await self._call_with_retries(
                    response_id,
                    target,
                    request,
                    plan.retries,
                    context,
                    deadline_at,
                )
            except RetryableUpstreamError:
                continue
            if self._valid_json(outcome.content):
                return outcome
        raise ExecutionFailed("cascade exhausted without valid JSON")

    async def _fan_out(
        self,
        response_id: str,
        request: ResponseRequest,
        plan: ExecutionPlan,
        context: ExecutionContext,
        deadline_at: float,
    ) -> AttemptOutcome:
        branches = [
            asyncio.create_task(
                self._call_with_retries(
                    response_id,
                    self.registry.by_id[model_id],
                    request,
                    plan.retries,
                    context,
                    deadline_at,
                ),
                name=f"fan-out-{response_id}-{model_id}",
            )
            for model_id in plan.models
        ]
        context.tasks.update(branches)
        selected: AttemptOutcome | None = None
        try:
            for completed in asyncio.as_completed(branches):
                try:
                    outcome = await completed
                except (RetryableUpstreamError, FatalUpstreamError):
                    continue
                if self._valid_json(outcome.content):
                    selected = outcome
                    break
        finally:
            for task in branches:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*branches, return_exceptions=True)
            context.tasks.difference_update(branches)
        if not selected:
            raise ExecutionFailed("fan-out produced no valid JSON")
        return selected

    async def _call_with_retries(
        self,
        response_id: str,
        target: ModelTarget,
        request: ResponseRequest,
        retries: int,
        context: ExecutionContext,
        deadline_at: float,
    ) -> AttemptOutcome:
        last_error: RetryableUpstreamError | None = None
        for _ in range(retries + 1):
            try:
                return await self._invoke_once(
                    response_id,
                    target,
                    request,
                    context,
                    deadline_at,
                )
            except RetryableUpstreamError as exc:
                last_error = exc
        assert last_error is not None
        raise last_error

    async def _invoke_once(
        self,
        response_id: str,
        target: ModelTarget,
        request: ResponseRequest,
        context: ExecutionContext,
        deadline_at: float,
    ) -> AttemptOutcome:
        if context.cancellation.is_set():
            raise asyncio.CancelledError
        remaining = deadline_at - asyncio.get_running_loop().time()
        if remaining <= 0:
            raise RetryableUpstreamError("end-to-end deadline exceeded")

        attempt_id = self.database.start_attempt(
            response_id,
            target.id,
            context.next_attempt(),
        )
        started = time.perf_counter()
        call = asyncio.create_task(
            self.upstream.generate(
                target,
                request,
                min(remaining, self.settings.default_timeout_ms / 1000),
            ),
            name=f"upstream-{response_id}-{target.id}",
        )
        cancelled = asyncio.create_task(context.cancellation.wait())
        context.tasks.update({call, cancelled})
        try:
            done, _ = await asyncio.wait(
                {call, cancelled},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if cancelled in done and context.cancellation.is_set():
                call.cancel()
                with suppress(asyncio.CancelledError):
                    await call
                raise asyncio.CancelledError
            if call not in done:
                call.cancel()
                with suppress(asyncio.CancelledError):
                    await call
                raise RetryableUpstreamError("end-to-end deadline exceeded")
            result = await call
            cost = (
                result.input_tokens * target.input_price_per_1k_tokens
                + result.output_tokens * target.output_price_per_1k_tokens
            ) / 1000
            self.database.finish_attempt(
                attempt_id,
                "completed",
                (time.perf_counter() - started) * 1000,
                result.input_tokens,
                result.output_tokens,
                cost,
            )
            return AttemptOutcome(
                model_id=target.id,
                content=result.content,
                input_tokens=result.input_tokens,
                output_tokens=result.output_tokens,
                cost_usd=cost,
            )
        except asyncio.CancelledError:
            self.database.finish_attempt(
                attempt_id,
                "cancelled",
                (time.perf_counter() - started) * 1000,
            )
            raise
        except RetryableUpstreamError as exc:
            self.database.finish_attempt(
                attempt_id,
                "retryable_error",
                (time.perf_counter() - started) * 1000,
                error=str(exc),
            )
            raise
        except Exception as exc:
            self.database.finish_attempt(
                attempt_id,
                "failed",
                (time.perf_counter() - started) * 1000,
                error=str(exc),
            )
            if isinstance(exc, FatalUpstreamError):
                raise
            raise FatalUpstreamError(str(exc)) from exc
        finally:
            cancelled.cancel()
            with suppress(asyncio.CancelledError):
                await cancelled
            context.tasks.discard(call)
            context.tasks.discard(cancelled)

    @staticmethod
    def _valid_json(content: str) -> bool:
        try:
            json.loads(content)
        except (TypeError, json.JSONDecodeError):
            return False
        return True

    async def shutdown(self) -> None:
        tasks: list[asyncio.Task] = []
        for context in self.active.values():
            context.cancellation.set()
            tasks.extend(task for task in context.tasks if not task.done())
            for task in context.tasks:
                if not task.done():
                    task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        await self.upstream.close()
        self.database.close()
