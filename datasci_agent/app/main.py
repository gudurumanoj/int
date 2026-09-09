from __future__ import annotations

import asyncio
import json
from typing import Any

from fastapi import BackgroundTasks, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from .config import Settings
from .db import ConflictError, Database, NotFoundError, TERMINAL_STATUSES
from .llm import DataScienceLLM
from .models import (
    ApprovalDecisionRequest,
    CancelRequest,
    CreateRunRequest,
    CreateSessionRequest,
    RegisterDatasetRequest,
)
from .orchestrator import Orchestrator
from .runner import DockerRunner
from .storage import Storage


def create_app(
    settings: Settings | None = None,
    llm: Any | None = None,
    runner: Any | None = None,
    *,
    auto_execute: bool = True,
) -> FastAPI:
    settings = settings or Settings.from_env()
    storage = Storage(settings.data_dir)
    db = Database(settings.data_dir / "agent.sqlite3")
    db.initialize()
    llm = llm or DataScienceLLM(settings)
    runner = runner or DockerRunner(settings, storage)
    orchestrator = Orchestrator(db, storage, llm, runner)

    api = FastAPI(
        title="Standalone Data-Science Agent Demo",
        version="0.1.0",
        description="Educational planner-worker agent with durable orchestration.",
    )
    api.state.settings = settings
    api.state.storage = storage
    api.state.db = db
    api.state.llm = llm
    api.state.runner = runner
    api.state.orchestrator = orchestrator
    api.state.auto_execute = auto_execute

    @api.exception_handler(ConflictError)
    async def conflict_handler(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @api.exception_handler(NotFoundError)
    async def not_found_handler(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @api.post("/v1/datasci/sessions")
    async def create_session(
        body: CreateSessionRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> JSONResponse:
        status, response, _ = db.create_session(
            idempotency_key, body.model_dump(mode="json")
        )
        return JSONResponse(status_code=status, content=response)

    @api.post("/v1/datasci/datasets", status_code=201)
    async def register_dataset(body: RegisterDatasetRequest) -> dict[str, Any]:
        try:
            stored = storage.store_dataset(body.filename, body.content)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        record = db.register_dataset(**stored)
        return {
            "dataset_id": record["dataset_id"],
            "version": record["version"],
            "sha256": record["sha256"],
            "filename": record["filename"],
        }

    @api.post("/v1/datasci/sessions/{session_id}/runs")
    async def create_run(
        session_id: str,
        body: CreateRunRequest,
        background_tasks: BackgroundTasks,
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> JSONResponse:
        payload = body.model_dump(mode="json")
        for item in payload["datasets"]:
            db.get_dataset(item["dataset_id"], item["version"], item["sha256"])
        status, response, replay = db.create_run(
            session_id, idempotency_key, payload
        )
        if auto_execute and not replay:
            background_tasks.add_task(orchestrator.execute_run, response["run_id"])
        return JSONResponse(status_code=status, content=response)

    @api.get("/v1/datasci/runs/{run_id}")
    async def get_run(run_id: str) -> dict[str, Any]:
        run = db.get_run(run_id)
        return {
            "run_id": run["id"],
            "session_id": run["session_id"],
            "status": run["status"],
            "question": run["question"],
            "datasets": run["datasets"],
            "config": run["config"],
            "current_plan_version": run["current_plan_version"],
            "cancellation_requested": run["cancellation_requested"],
            "report_artifact_id": run["report_artifact_id"],
            "manifest_artifact_id": run["manifest_artifact_id"],
            "error": run["error"],
            "artifacts": [
                {
                    "artifact_id": item["id"],
                    "kind": item["kind"],
                    "path": item["path"],
                    "sha256": item["sha256"],
                    "partial": bool(item["partial"]),
                }
                for item in db.list_artifacts(run_id)
            ],
        }

    @api.post("/v1/datasci/runs/{run_id}/cancel")
    async def cancel_run(
        run_id: str,
        body: CancelRequest,
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> JSONResponse:
        status, response, replay = db.request_cancel(
            run_id, idempotency_key, body.model_dump(mode="json")
        )
        if not replay:
            await orchestrator.cancel(run_id)
        return JSONResponse(status_code=status, content=response)

    @api.put("/v1/datasci/approval-requests/{approval_id}/decision")
    async def decide_approval(
        approval_id: str,
        body: ApprovalDecisionRequest,
        background_tasks: BackgroundTasks,
        idempotency_key: str = Header(alias="Idempotency-Key"),
    ) -> JSONResponse:
        status, response, replay = db.decide_approval(
            approval_id, idempotency_key, body.model_dump(mode="json")
        )
        if (
            auto_execute
            and not replay
            and response["decision"] in {"approve", "request_changes"}
        ):
            background_tasks.add_task(
                orchestrator.execute_run, response["run_id"]
            )
        return JSONResponse(status_code=status, content=response)

    @api.get("/v1/datasci/runs/{run_id}/events")
    async def events(
        request: Request,
        run_id: str,
        last_event_id: str | None = Header(default=None, alias="Last-Event-ID"),
    ) -> StreamingResponse:
        try:
            after = int(last_event_id or "0")
        except ValueError as exc:
            raise HTTPException(
                status_code=400, detail="Last-Event-ID must be an integer"
            ) from exc
        if after < 0:
            raise HTTPException(status_code=400, detail="Last-Event-ID must be >= 0")
        db.get_run(run_id)

        async def stream():
            cursor = after
            while True:
                batch = db.get_events(run_id, cursor)
                for event in batch:
                    cursor = event["seq"]
                    data = json.dumps(
                        {
                            "type": event["event_type"],
                            "data": event["data"],
                            "created_at": event["created_at"],
                        },
                        separators=(",", ":"),
                    )
                    yield f"id: {cursor}\nevent: {event['event_type']}\ndata: {data}\n\n"
                run = db.get_run(run_id)
                if run["status"] in TERMINAL_STATUSES and not batch:
                    break
                if await request.is_disconnected():
                    break
                if not batch:
                    yield ": keep-alive\n\n"
                await asyncio.sleep(0.1)

        return StreamingResponse(
            stream(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return api


app = create_app()
