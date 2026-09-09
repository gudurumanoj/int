from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from .catalog import Catalog
from .config import Settings
from .db import Database
from .llm import OpenAICompatibleReranker, Reranker
from .models import FeedbackRequest, RecommendationRequest
from .service import (
    ConflictError,
    InvalidFeedbackError,
    NotFoundError,
    RecommendationService,
)


def create_app(
    settings: Settings | None = None,
    reranker: Reranker | None = None,
) -> FastAPI:
    settings = settings or Settings.from_env()
    database = Database(settings.database_path)
    database.initialize()
    catalog = Catalog(settings.catalog_path)
    if reranker is None and settings.llm_enabled:
        reranker = OpenAICompatibleReranker(settings)
    service = RecommendationService(
        settings=settings,
        database=database,
        catalog=catalog,
        reranker=reranker,
    )

    app = FastAPI(
        title="LLM-Assisted Recommendation Demo",
        version="0.1.0",
        description=(
            "Static eligibility and token-overlap ranking with optional, "
            "strictly validated LLM reranking."
        ),
    )
    app.state.settings = settings
    app.state.database = database
    app.state.service = service

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/recommendations")
    async def recommendations(
        request: RecommendationRequest,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=200)
        ],
        tenant_id: Annotated[
            str | None, Header(alias="X-Tenant-ID", max_length=100)
        ] = None,
    ) -> JSONResponse:
        resolved_tenant = tenant_id or settings.default_tenant_id
        try:
            response, status_code = await service.recommend(
                tenant_id=resolved_tenant,
                idempotency_key=idempotency_key,
                request=request,
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return JSONResponse(response, status_code=status_code)

    @app.post("/v1/recommendations/{set_id}/feedback")
    async def feedback(
        set_id: str,
        request: FeedbackRequest,
        idempotency_key: Annotated[
            str, Header(alias="Idempotency-Key", min_length=1, max_length=200)
        ],
        tenant_id: Annotated[
            str | None, Header(alias="X-Tenant-ID", max_length=100)
        ] = None,
    ) -> JSONResponse:
        resolved_tenant = tenant_id or settings.default_tenant_id
        try:
            response, status_code = await service.feedback(
                tenant_id=resolved_tenant,
                set_id=set_id,
                idempotency_key=idempotency_key,
                request=request,
            )
        except ConflictError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        except NotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc
        except InvalidFeedbackError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return JSONResponse(response, status_code=status_code)

    return app


app = create_app()
