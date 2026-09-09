from __future__ import annotations

import math

from app.config import Settings
from app.domain import ExecutionPlan, ModelRegistry, ModelTarget, ResponseRequest


class RoutingError(Exception):
    def __init__(self, status_code: int, message: str):
        super().__init__(message)
        self.status_code = status_code
        self.message = message


class Router:
    _tier = {"low": 0, "balanced": 1, "high": 2}

    def __init__(self, registry: ModelRegistry, settings: Settings):
        self.registry = registry
        self.settings = settings

    def plan(self, request: ResponseRequest) -> ExecutionPlan:
        task = "extraction" if request.response_format == "json" else request.routing.task_type
        if task == "auto":
            task = "chat"

        candidates = [
            target
            for target in self.registry.models
            if target.health == "healthy" and task in target.capabilities
        ]
        if request.routing.model_override:
            override = self.registry.by_id.get(request.routing.model_override)
            if override not in candidates:
                raise RoutingError(503, "model override is unhealthy or incompatible")
            candidates = [override]
        if not candidates:
            raise RoutingError(503, "no healthy model has the required capability")

        desired = self._tier[request.routing.quality]
        primary = min(
            candidates,
            key=lambda model: (
                abs(self._tier[model.quality_tier] - desired),
                self._model_cost(model, request),
            ),
        )
        strategy = request.routing.strategy
        if request.routing.model_override:
            strategy = "direct"
        elif strategy == "auto":
            # Without a deterministic task gate, quality tiers are routing targets,
            # not confidence estimates. Auto routing therefore stays direct.
            strategy = "direct"

        gate_available = (
            request.response_format == "json"
            and any(model.quality_gate == "json_valid" for model in candidates)
        )
        if strategy in {"cascade", "fan_out"} and not gate_available:
            strategy = "direct"

        ordered = sorted(
            candidates,
            key=lambda model: (self._tier[model.quality_tier], self._model_cost(model, request)),
        )
        if strategy == "cascade":
            selected = [
                model
                for model in ordered
                if model.quality_gate == "json_valid"
                and self._tier[model.quality_tier] <= desired
            ]
            if not selected:
                selected = [primary]
            if primary not in selected:
                selected.append(primary)
            selected = list({model.id: model for model in selected}.values())
            if len(selected) == 1:
                strategy = "direct"
        elif strategy == "fan_out":
            selected = [model for model in ordered if model.quality_gate == "json_valid"]
            if len(selected) < 2:
                selected = [primary]
                strategy = "direct"
        else:
            fallbacks = sorted(
                (model for model in candidates if model != primary),
                key=lambda model: (
                    abs(self._tier[model.quality_tier] - desired),
                    self._model_cost(model, request),
                ),
            )
            selected = [primary, *fallbacks]

        attempts = self.settings.default_retries + 1
        call_costs = [self._model_cost(model, request) for model in selected]
        estimated_cost = attempts * sum(call_costs)
        if strategy == "fan_out":
            estimated_latency = attempts * max(model.p99_latency_ms for model in selected)
        else:
            estimated_latency = attempts * sum(model.p99_latency_ms for model in selected)

        deadline_ms = request.routing.max_latency_ms or self.settings.default_timeout_ms
        if request.routing.max_cost_usd is not None and estimated_cost > request.routing.max_cost_usd:
            raise RoutingError(
                422,
                f"worst-case cost ${estimated_cost:.6f} exceeds request budget",
            )
        if estimated_latency > deadline_ms:
            raise RoutingError(
                422,
                f"estimated p99 latency {estimated_latency:.0f}ms exceeds deadline",
            )

        return ExecutionPlan(
            strategy=strategy,
            models=[model.id for model in selected],
            retries=self.settings.default_retries,
            quality_gate_id="json_valid" if gate_available else None,
            registry_version=self.registry.version,
            estimated_max_cost_usd=round(estimated_cost, 9),
            estimated_p99_latency_ms=estimated_latency,
            deadline_ms=deadline_ms,
        )

    def _model_cost(self, model: ModelTarget, request: ResponseRequest) -> float:
        input_tokens = max(
            1,
            math.ceil(sum(len(message.content) for message in request.messages) / 4),
        )
        max_tokens = request.max_tokens or self.settings.default_max_tokens
        return (
            input_tokens * model.input_price_per_1k_tokens
            + max_tokens * model.output_price_per_1k_tokens
        ) / 1000
