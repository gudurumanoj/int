from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from typing import Any

from app.config import Settings
from app.llm import ChatClient
from app.models import ToolResult
from app.storage import (
    Conflict,
    Storage,
    TERMINAL_STATUSES,
    canonical_json,
    utc_now,
)
from app.tools import SandboxRunner, ToolRegistry


SYSTEM_PROMPT = """You are an educational code agent working in one confined workspace.
Use structured tool calls to inspect and change files. Shell calls execute in a
networkless container and require human approval. Keep assistant text concise
and user-visible. Never reveal or request hidden chain-of-thought; return only
short progress summaries, tool calls, and a final answer.
"""


class AgentRuntime:
    def __init__(
        self,
        storage: Storage,
        client: ChatClient,
        tools: ToolRegistry,
        runner: SandboxRunner,
        settings: Settings,
    ):
        self.storage = storage
        self.client = client
        self.tools = tools
        self.runner = runner
        self.settings = settings
        self._approval_events: dict[str, asyncio.Event] = {}
        self._cancel_locks: dict[str, asyncio.Lock] = {}

    async def execute_run(self, run_id: str) -> None:
        if not self.storage.transition_to_running(run_id):
            return
        run = self.storage.get_run(run_id)
        try:
            async with asyncio.timeout(
                float(run["config"]["max_wall_time_seconds"])
            ):
                await self._execute_steps(run)
        except TimeoutError:
            self.storage.fail_run(run_id, "Run wall-time budget exhausted")
        except asyncio.CancelledError:
            current = self.storage.get_run(run_id)
            if current["status"] not in TERMINAL_STATUSES:
                self.storage.request_cancel(run_id, "Service is shutting down")
                await self.runner.cancel(run_id)
                self.storage.mark_cancelled(run_id)
            raise
        except Exception as exc:
            self.storage.fail_run(run_id, f"{type(exc).__name__}: {exc}")

    async def _execute_steps(self, run: dict[str, Any]) -> None:
        run_id = run["run_id"]
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": run["input"]},
        ]
        max_steps = int(run["config"]["max_steps"])
        model = run["config"].get("model")
        for _step in range(max_steps):
            if self.storage.get_run(run_id)["status"] != "running":
                return
            turn = await self.client.complete(
                messages,
                self.tools.definitions,
                model=model,
            )
            if self.storage.get_run(run_id)["status"] != "running":
                return
            if turn.summary:
                self.storage.append_event(
                    run_id,
                    "assistant.progress",
                    {"run_id": run_id, "summary": turn.summary},
                )
            if not turn.tool_calls:
                self.storage.complete_run(run_id, turn.summary)
                return

            assistant_calls = []
            for call in turn.tool_calls:
                if not call.id:
                    raise ValueError("Tool call ID must not be empty")
                self.tools.validate(call.name, call.arguments)
                assistant_calls.append(
                    {
                        "id": call.id,
                        "type": "function",
                        "function": {
                            "name": call.name,
                            "arguments": canonical_json(call.arguments),
                        },
                    }
                )
            messages.append(
                {
                    "role": "assistant",
                    "content": turn.summary or None,
                    "tool_calls": assistant_calls,
                }
            )
            for call in turn.tool_calls:
                if self.storage.get_run(run_id)["status"] != "running":
                    return
                result = await self._dispatch_tool(
                    run_id, call.id, call.name, call.arguments
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": result.model_dump_json(),
                    }
                )
        self.storage.fail_run(run_id, "Step budget exhausted")

    async def _dispatch_tool(
        self,
        run_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> ToolResult:
        intent, replayed = self.storage.begin_tool_call(
            run_id, tool_call_id, name, arguments
        )
        if replayed:
            if intent["result"] is not None:
                return ToolResult.model_validate(intent["result"])
            raise Conflict(
                "A prior tool call has an ambiguous unfinished outcome"
            )

        if self.tools.requires_approval(name):
            expires_at = (
                datetime.now(UTC)
                + timedelta(seconds=self.settings.approval_ttl_seconds)
            ).isoformat()
            approval = self.storage.create_approval(
                run_id,
                tool_call_id,
                arguments,
                "Shell commands execute arbitrary code in the Docker sandbox.",
                expires_at,
            )
            decision = await self._wait_for_approval(
                approval["approval_id"], expires_at
            )
            if decision == "deny":
                now = utc_now()
                result = ToolResult(
                    tool_call_id=tool_call_id,
                    tool_name=name,
                    status="failed",
                    stderr="Human approval denied this tool call",
                    error_code="approval_denied",
                    started_at=now,
                    finished_at=now,
                )
                self.storage.finish_tool_call(
                    run_id, tool_call_id, result.model_dump()
                )
                return result

        if self.storage.get_run(run_id)["status"] != "running":
            now = utc_now()
            return ToolResult(
                tool_call_id=tool_call_id,
                tool_name=name,
                status="cancelled",
                stderr="Run was cancelled before tool execution",
                error_code="run_cancelled",
                started_at=now,
                finished_at=now,
            )
        self.storage.start_tool_call(run_id, tool_call_id, name)
        result = await self.tools.execute(
            run_id, tool_call_id, name, arguments
        )
        self.storage.finish_tool_call(
            run_id, tool_call_id, result.model_dump()
        )
        return result

    async def _wait_for_approval(
        self, approval_id: str, expires_at: str
    ) -> str:
        event = self._approval_events.setdefault(
            approval_id, asyncio.Event()
        )
        expiry = datetime.fromisoformat(expires_at)
        try:
            while True:
                approval = self.storage.get_approval(approval_id)
                if approval["status"] in {"allow_once", "deny"}:
                    return str(approval["status"])
                run = self.storage.get_run(approval["run_id"])
                if run["status"] in TERMINAL_STATUSES | {"cancelling"}:
                    return "deny"
                remaining = (expiry - datetime.now(UTC)).total_seconds()
                if remaining <= 0:
                    raise TimeoutError("Approval request expired")
                try:
                    await asyncio.wait_for(
                        event.wait(), timeout=min(remaining, 0.5)
                    )
                except TimeoutError:
                    continue
                event.clear()
        finally:
            self._approval_events.pop(approval_id, None)

    def decide_approval(
        self,
        approval_id: str,
        idempotency_key: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        response = self.storage.decide_approval(
            approval_id, idempotency_key, payload
        )
        event = self._approval_events.get(approval_id)
        if event is not None:
            event.set()
        return response

    async def cancel_run(
        self,
        run_id: str,
        idempotency_key: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        lock = self._cancel_locks.setdefault(run_id, asyncio.Lock())
        scope = f"POST:/v1/agent-runs/{run_id}/cancel"
        async with lock:
            existing = self.storage.lookup_idempotency(
                scope, idempotency_key, payload
            )
            if existing is not None:
                return existing[0], True
            requested = self.storage.request_cancel(run_id, payload["reason"])
            if requested["status"] == "cancelling":
                await self.runner.cancel(run_id)
                response = self.storage.mark_cancelled(run_id)
            else:
                response = requested
            self.storage.save_idempotency(
                scope, idempotency_key, payload, response, 200
            )
            return response, False
