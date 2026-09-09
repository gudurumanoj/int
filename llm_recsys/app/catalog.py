from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from .config import Settings
from .models import CatalogItem, RecommendationRequest


TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = {
    "a",
    "an",
    "and",
    "for",
    "in",
    "of",
    "on",
    "the",
    "to",
    "with",
}


def tokenize(value: str) -> set[str]:
    return {
        token
        for token in TOKEN_RE.findall(value.lower())
        if len(token) > 1 and token not in STOP_WORDS
    }


@dataclass(frozen=True, slots=True)
class RankedCandidate:
    item: CatalogItem
    overlap: int
    matched_terms: tuple[str, ...]


class Catalog:
    def __init__(self, path: Path):
        raw_items = json.loads(path.read_text(encoding="utf-8"))
        self.items = [CatalogItem.model_validate(item) for item in raw_items]

    def eligible(
        self,
        tenant_id: str,
        request: RecommendationRequest,
        settings: Settings,
    ) -> list[CatalogItem]:
        allowed_categories = set(settings.eligibility_allowed_categories)
        eligible_items: list[CatalogItem] = []
        for item in self.items:
            if settings.eligibility_require_available and not item.available:
                continue
            if tenant_id not in item.tenants and "*" not in item.tenants:
                continue
            if request.placement not in item.placements:
                continue
            if (
                request.context.locale not in item.locales
                and "*" not in item.locales
            ):
                continue
            if allowed_categories and item.category.lower() not in allowed_categories:
                continue
            eligible_items.append(item)
        return eligible_items[: settings.eligibility_max_candidates]

    def rank(
        self, items: list[CatalogItem], request: RecommendationRequest
    ) -> list[RankedCandidate]:
        request_text = " ".join(
            [
                request.placement,
                request.context.query or "",
                *request.context.interests,
            ]
        )
        request_tokens = tokenize(request_text)
        ranked: list[RankedCandidate] = []
        for item in items:
            item_tokens = tokenize(
                " ".join(
                    [
                        item.title,
                        item.description,
                        item.category,
                        *item.tags,
                    ]
                )
            )
            matched = tuple(sorted(request_tokens & item_tokens))
            ranked.append(
                RankedCandidate(
                    item=item,
                    overlap=len(matched),
                    matched_terms=matched,
                )
            )
        return sorted(ranked, key=lambda candidate: (-candidate.overlap, candidate.item.item_id))


def deterministic_reason(
    candidate: RankedCandidate, request: RecommendationRequest
) -> str:
    if candidate.matched_terms:
        terms = ", ".join(candidate.matched_terms[:3])
        return (
            f"Matches your stated interest in {terms}, using this item's "
            "catalog metadata."
        )
    return (
        f"Available for the {request.placement} placement in "
        f"{request.context.locale}."
    )
