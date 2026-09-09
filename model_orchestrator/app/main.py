from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from typing import Annotated, AsyncIterator

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse

from app.config import Settings
from app.database import Database, EventHistoryGone, IdempotencyConflict
from app.domain import ModelRegistry, ResponseRequest, ResponseSnapshot
from app.routing import RoutingError
from app.service import Orchestrator
from app.upstream import OpenAIUpstream, Upstream


def create_app(
    settings: Settings | None = None,
    upstream: Upstream | None = None,
) -> FastAPI:
    settings = settings or Settings()
    registry = ModelRegistry(settings.model_registry_path)
    database = Database(
        settings.sqlite_path,
        settings.idempotency_ttl_seconds,
        settings.event_retention_seconds,
    )
    orchestrator = Orchestrator(
        settings,
        registry,
        database,
        upstream or OpenAIUpstream(settings),
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        await orchestrator.shutdown()

    app = FastAPI(
        title="Educational Model Orchestrator",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.orchestrator = orchestrator

    @app.post("/v1/responses", response_model=ResponseSnapshot)
    async def create_response(
        request: ResponseRequest,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=255)
        ],
        tenant_id: Annotated[
            str, Header(alias="X-Tenant-ID", min_length=1, max_length=255)
        ] = "demo",
    ) -> JSONResponse:
        try:
            snapshot, replayed = orchestrator.create(
                tenant_id,
                idempotency_key,
                request,
            )
        except IdempotencyConflict as exc:
            raise HTTPException(
                status_code=409,
                detail="idempotency key was already used with a different payload",
            ) from exc
        except RoutingError as exc:
            raise HTTPException(
                status_code=exc.status_code,
                detail=exc.message,
            ) from exc
        return JSONResponse(
            status_code=200 if replayed else 202,
            content=snapshot.model_dump(mode="json"),
            headers={"Idempotent-Replay": "true" if replayed else "false"},
        )

    @app.get("/v1/responses/{response_id}", response_model=ResponseSnapshot)
    async def get_response(response_id: str) -> ResponseSnapshot:
        snapshot = database.get(response_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="response not found")
        return snapshot

    @app.delete("/v1/responses/{response_id}", response_model=ResponseSnapshot)
    async def cancel_response(response_id: str) -> ResponseSnapshot:
        snapshot = await orchestrator.cancel(response_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="response not found")
        return snapshot

    @app.get("/v1/responses/{response_id}/events")
    async def response_events(
        response_id: str,
        last_event_id: Annotated[
            str | None, Header(alias="Last-Event-ID")
        ] = None,
    ) -> StreamingResponse:
        try:
            cursor = int(last_event_id) if last_event_id is not None else 0
            if cursor < 0:
                raise ValueError
        except ValueError as exc:
            raise HTTPException(
                status_code=400,
                detail="Last-Event-ID must be a non-negative integer",
            ) from exc

        try:
            initial = database.events_after(response_id, cursor)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail="response not found") from exc
        except EventHistoryGone as exc:
            raise HTTPException(status_code=410, detail="event history expired") from exc

        async def generate() -> AsyncIterator[str]:
            nonlocal cursor
            pending = initial
            while True:
                for event in pending:
                    cursor = event["id"]
                    payload = json.dumps(event["data"], separators=(",", ":"))
                    yield (
                        f"id: {event['id']}\n"
                        f"event: {event['event']}\n"
                        f"data: {payload}\n\n"
                    )
                snapshot = database.get(response_id)
                if snapshot and snapshot.status != "in_progress":
                    if cursor >= snapshot.event_cursor:
                        return
                await asyncio.sleep(0.02)
                try:
                    pending = database.events_after(response_id, cursor)
                except EventHistoryGone:
                    return

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    return app
