from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Header, Request
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings
from app.llm import ChatClient, OpenAIChatClient
from app.models import (
    ApprovalDecisionRequest,
    CancelRequest,
    RunCreate,
    SessionCreate,
)
from app.runtime import AgentRuntime
from app.storage import (
    Conflict,
    NotFound,
    Storage,
    TERMINAL_STATUSES,
    canonical_json,
)
from app.tools import DockerRunner, SandboxRunner, ToolRegistry


def create_app(
    settings: Settings | None = None,
    *,
    llm_client: ChatClient | None = None,
    storage: Storage | None = None,
    runner: SandboxRunner | None = None,
) -> FastAPI:
    configured = settings or Settings.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        store = storage or Storage(configured.db_path)
        store.initialize()
        configured.workspace_root.mkdir(parents=True, exist_ok=True)
        selected_runner = runner or DockerRunner(configured.docker_image)
        selected_client = llm_client or OpenAIChatClient(configured)
        tools = ToolRegistry(configured.workspace_root, selected_runner)
        runtime = AgentRuntime(
            store,
            selected_client,
            tools,
            selected_runner,
            configured,
        )
        app.state.storage = store
        app.state.runtime = runtime
        app.state.background_tasks = set()
        try:
            yield
        finally:
            tasks = list(app.state.background_tasks)
            for task in tasks:
                task.cancel()
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
            store.close()

    app = FastAPI(
        title="Educational Code Agent",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.exception_handler(NotFound)
    async def not_found_handler(
        _request: Request, exc: NotFound
    ) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(Conflict)
    async def conflict_handler(
        _request: Request, exc: Conflict
    ) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    def start_run(request: Request, run_id: str) -> None:
        task = asyncio.create_task(
            request.app.state.runtime.execute_run(run_id),
            name=f"agent-run:{run_id}",
        )
        tasks: set[asyncio.Task[None]] = (
            request.app.state.background_tasks
        )
        tasks.add(task)
        task.add_done_callback(tasks.discard)

    @app.post("/v1/agent-sessions")
    async def create_session(
        body: SessionCreate,
        request: Request,
        idempotency_key: str = Header(
            min_length=1, alias="Idempotency-Key"
        ),
    ) -> JSONResponse:
        response, status_code, _ = request.app.state.storage.create_session(
            idempotency_key, body.model_dump(mode="json")
        )
        return JSONResponse(status_code=status_code, content=response)

    @app.post("/v1/agent-sessions/{session_id}/runs")
    async def create_agent_run(
        session_id: str,
        body: RunCreate,
        request: Request,
        idempotency_key: str = Header(
            min_length=1, alias="Idempotency-Key"
        ),
    ) -> JSONResponse:
        response, status_code, replayed = (
            request.app.state.storage.create_run(
                session_id,
                idempotency_key,
                body.model_dump(mode="json"),
            )
        )
        if not replayed:
            start_run(request, response["run_id"])
        return JSONResponse(status_code=status_code, content=response)

    @app.get("/v1/agent-runs/{run_id}")
    async def get_agent_run(run_id: str, request: Request) -> dict[str, Any]:
        run = request.app.state.storage.get_run(run_id)
        return {
            "run_id": run["run_id"],
            "session_id": run["session_id"],
            "status": run["status"],
            "result": run["result"],
            "terminal_reason": run["terminal_reason"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
        }

    @app.post("/v1/agent-runs/{run_id}/cancel")
    async def cancel_agent_run(
        run_id: str,
        body: CancelRequest,
        request: Request,
        idempotency_key: str = Header(
            min_length=1, alias="Idempotency-Key"
        ),
    ) -> dict[str, Any]:
        response, _ = await request.app.state.runtime.cancel_run(
            run_id,
            idempotency_key,
            body.model_dump(mode="json"),
        )
        return response

    @app.put("/v1/approval-requests/{approval_id}/decision")
    async def decide_approval(
        approval_id: str,
        body: ApprovalDecisionRequest,
        request: Request,
        idempotency_key: str = Header(
            min_length=1, alias="Idempotency-Key"
        ),
    ) -> dict[str, Any]:
        response, _ = request.app.state.runtime.decide_approval(
            approval_id,
            idempotency_key,
            body.model_dump(mode="json"),
        )
        return response

    @app.get("/v1/agent-runs/{run_id}/events")
    async def get_agent_events(
        run_id: str,
        request: Request,
        last_event_id: str | None = Header(
            default=None, alias="Last-Event-ID"
        ),
    ) -> StreamingResponse:
        request.app.state.storage.get_run(run_id)
        try:
            cursor = int(last_event_id) if last_event_id is not None else 0
        except ValueError:
            return JSONResponse(  # type: ignore[return-value]
                status_code=400,
                content={"detail": "Last-Event-ID must be an integer"},
            )
        if cursor < 0:
            return JSONResponse(  # type: ignore[return-value]
                status_code=400,
                content={"detail": "Last-Event-ID must not be negative"},
            )

        async def stream() -> AsyncIterator[str]:
            nonlocal cursor
            last_heartbeat = time.monotonic()
            while True:
                events = request.app.state.storage.list_events_after(
                    run_id, cursor
                )
                for event in events:
                    cursor = event["sequence"]
                    yield (
                        f"id: {cursor}\n"
                        f"event: {event['event_type']}\n"
                        f"data: {canonical_json(event['data'])}\n\n"
                    )
                run = request.app.state.storage.get_run(run_id)
                if run["status"] in TERMINAL_STATUSES and not events:
                    return
                now = time.monotonic()
                if (
                    now - last_heartbeat
                    >= configured.sse_heartbeat_seconds
                ):
                    yield ": heartbeat\n\n"
                    last_heartbeat = now
                await asyncio.sleep(0.25)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    return app


app = create_app()
