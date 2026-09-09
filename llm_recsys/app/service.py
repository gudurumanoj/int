from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

from .catalog import Catalog, RankedCandidate, deterministic_reason, tokenize
from .config import Settings
from .db import Database
from .llm import Reranker
from .models import FeedbackRequest, LLMRerankResponse, RecommendationRequest
from .security import sign_tracking_token, verify_tracking_token


class ConflictError(Exception):
    pass


class NotFoundError(Exception):
    pass


class InvalidFeedbackError(Exception):
    pass


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def isoformat(value: datetime) -> str:
    return value.isoformat(timespec="milliseconds").replace("+00:00", "Z")


def canonical_hash(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode(
        "utf-8"
    )
    return hashlib.sha256(encoded).hexdigest()


class RecommendationService:
    def __init__(
        self,
        *,
        settings: Settings,
        database: Database,
        catalog: Catalog,
        reranker: Reranker | None,
    ):
        self.settings = settings
        self.database = database
        self.catalog = catalog
        self.reranker = reranker
        self._write_lock = asyncio.Lock()

    async def recommend(
        self,
        *,
        tenant_id: str,
        idempotency_key: str,
        request: RecommendationRequest,
    ) -> tuple[dict[str, Any], int]:
        request_json = request.model_dump(mode="json")
        request_hash = canonical_hash(request_json)
        async with self._write_lock:
            now = utc_now()
            now_text = isoformat(now)
            existing = self.database.get_idempotency(
                tenant_id, "recommendations", idempotency_key, now_text
            )
            if existing:
                if existing["request_hash"] != request_hash:
                    raise ConflictError(
                        "Idempotency-Key was already used with a different payload"
                    )
                return existing["response"], existing["status_code"]

            eligible = self.catalog.eligible(tenant_id, request, self.settings)
            deterministic = self.catalog.rank(eligible, request)
            ordered = deterministic
            llm_reasons: dict[str, str] = {}
            route = "deterministic"
            degraded = False

            shortlist = deterministic[: self.settings.llm_shortlist_size]
            if request.explain and self.reranker is not None and shortlist:
                try:
                    raw_result = await asyncio.wait_for(
                        self.reranker.rerank(
                            request,
                            shortlist,
                            min(request.num_results, len(shortlist)),
                        ),
                        timeout=self.settings.openai_timeout_seconds,
                    )
                    ordered, llm_reasons = self._validated_llm_order(
                        raw_result, shortlist, deterministic, request.num_results
                    )
                    route = "deterministic_plus_llm"
                except Exception:
                    ordered = deterministic
                    llm_reasons = {}
                    route = "deterministic_fallback"
                    degraded = True

            generated_at = now_text
            expires_at = isoformat(
                now + timedelta(seconds=self.settings.recommendation_ttl_seconds)
            )
            set_id = f"rec_{uuid.uuid4().hex}"
            recommendations: list[dict[str, Any]] = []
            for position, candidate in enumerate(
                ordered[: request.num_results], start=1
            ):
                item = candidate.item
                reason = self._grounded_reason(
                    candidate, request, llm_reasons.get(item.item_id)
                )
                recommendations.append(
                    {
                        "item_id": item.item_id,
                        "title": item.title,
                        "reason": reason,
                        "rank": position,
                        "tracking_token": sign_tracking_token(
                            secret=self.settings.hmac_secret,
                            tenant_id=tenant_id,
                            set_id=set_id,
                            item_id=item.item_id,
                            position=position,
                            user_id=request.user_id,
                            session_id=request.session_id,
                            expires_at=expires_at,
                        ),
                        "retrieval_sources": ["token_overlap"],
                    }
                )
            response = {
                "request_id": request.request_id,
                "recommendation_set_id": set_id,
                "user_id": request.user_id,
                "generated_at": generated_at,
                "expires_at": expires_at,
                "recommendations": recommendations,
                "metadata": {"route": route, "degraded": degraded},
            }
            idempotency_expires_at = isoformat(
                now + timedelta(seconds=self.settings.idempotency_ttl_seconds)
            )
            self.database.store_recommendation(
                tenant_id=tenant_id,
                idem_key=idempotency_key,
                request_hash=request_hash,
                request_json=request_json,
                response=response,
                idempotency_expires_at=idempotency_expires_at,
            )
            return response, 200

    @staticmethod
    def _validated_llm_order(
        raw_result: object,
        shortlist: list[RankedCandidate],
        deterministic: list[RankedCandidate],
        num_results: int,
    ) -> tuple[list[RankedCandidate], dict[str, str]]:
        result = LLMRerankResponse.model_validate(raw_result)
        if len(result.items) > min(num_results, len(shortlist)):
            raise ValueError("LLM returned too many items")
        ranks = sorted(item.rank for item in result.items)
        if ranks != list(range(1, len(result.items) + 1)):
            raise ValueError("LLM ranks must be contiguous and start at one")
        item_ids = [item.item_id for item in result.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("LLM returned duplicate item IDs")
        allowed = {candidate.item.item_id: candidate for candidate in shortlist}
        if any(item_id not in allowed for item_id in item_ids):
            raise ValueError("LLM returned an item outside the shortlist")

        llm_items = sorted(result.items, key=lambda item: item.rank)
        selected = [allowed[item.item_id] for item in llm_items]
        selected_ids = {candidate.item.item_id for candidate in selected}
        ordered = selected + [
            candidate
            for candidate in deterministic
            if candidate.item.item_id not in selected_ids
        ]
        reasons = {item.item_id: item.reason for item in llm_items}
        return ordered, reasons

    @staticmethod
    def _grounded_reason(
        candidate: RankedCandidate,
        request: RecommendationRequest,
        llm_reason: str | None,
    ) -> str:
        if llm_reason:
            reason_tokens = tokenize(llm_reason)
            item_tokens = tokenize(
                " ".join(
                    [
                        candidate.item.title,
                        candidate.item.description,
                        candidate.item.category,
                        *candidate.item.tags,
                    ]
                )
            )
            preference_tokens = tokenize(
                " ".join(
                    [request.context.query or "", *request.context.interests]
                )
            )
            has_item_evidence = bool(reason_tokens & item_tokens)
            has_preference_evidence = not preference_tokens or bool(
                reason_tokens & preference_tokens
            )
            if has_item_evidence and has_preference_evidence:
                return llm_reason
        return deterministic_reason(candidate, request)

    async def feedback(
        self,
        *,
        tenant_id: str,
        set_id: str,
        idempotency_key: str,
        request: FeedbackRequest,
    ) -> tuple[dict[str, Any], int]:
        request_json = request.model_dump(mode="json")
        hash_payload = {"set_id": set_id, "feedback": request_json}
        request_hash = canonical_hash(hash_payload)
        async with self._write_lock:
            now = utc_now()
            now_text = isoformat(now)
            existing_idempotency = self.database.get_idempotency(
                tenant_id, "feedback", idempotency_key, now_text
            )
            if existing_idempotency:
                if existing_idempotency["request_hash"] != request_hash:
                    raise ConflictError(
                        "Idempotency-Key was already used with a different payload"
                    )
                return (
                    existing_idempotency["response"],
                    existing_idempotency["status_code"],
                )

            recommendation_set = self.database.get_recommendation_set(
                tenant_id, set_id
            )
            if recommendation_set is None:
                raise NotFoundError("recommendation set not found")
            item = self.database.get_recommendation_item(
                tenant_id, set_id, request.item_id
            )
            if item is None:
                raise InvalidFeedbackError(
                    "tracking token does not match the recommendation item"
                )
            expires_at = recommendation_set["expires_at"]
            parsed_expiry = datetime.fromisoformat(
                expires_at.replace("Z", "+00:00")
            )
            if now >= parsed_expiry:
                raise InvalidFeedbackError("tracking token has expired")
            token_is_valid = verify_tracking_token(
                request.tracking_token,
                secret=self.settings.hmac_secret,
                tenant_id=tenant_id,
                set_id=set_id,
                item_id=request.item_id,
                position=request.position,
                user_id=recommendation_set["user_id"],
                session_id=request.session_id,
                expires_at=expires_at,
            )
            token_is_original = hmac.compare_digest(
                request.tracking_token, item["tracking_token"]
            )
            fields_match = (
                request.position == item["position"]
                and request.session_id == recommendation_set["session_id"]
            )
            if not token_is_valid or not token_is_original or not fields_match:
                raise InvalidFeedbackError(
                    "tracking token does not match the supplied feedback"
                )

            existing_event = self.database.get_feedback(
                tenant_id, request.event_id
            )
            if (
                existing_event is not None
                and existing_event["request_hash"] != request_hash
            ):
                raise ConflictError(
                    "event_id was already used with a different payload"
                )

            duplicate = existing_event is not None
            received_at = (
                existing_event["received_at"] if duplicate else now_text
            )
            response = {
                "event_id": request.event_id,
                "accepted": True,
                "deduplicated": duplicate,
                "received_at": received_at,
            }
            status_code = 200 if duplicate else 201
            event = None
            if not duplicate:
                event = {
                    **request_json,
                    "set_id": set_id,
                }
            self.database.store_feedback(
                tenant_id=tenant_id,
                idem_key=idempotency_key,
                request_hash=request_hash,
                event=event,
                response=response,
                created_at=now_text,
                idempotency_expires_at=isoformat(
                    now
                    + timedelta(seconds=self.settings.idempotency_ttl_seconds)
                ),
                status_code=status_code,
            )
            return response, status_code
