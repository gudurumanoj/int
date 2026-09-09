# Design: Intelligent Model Orchestrator

## 1. Problem Framing

> "Design a system that intelligently routes incoming requests to the right model (or chain of models), manages model lifecycle, and exposes a single unified API to consumers — while optimizing for cost, latency, and quality."

**Tower context**: The Core AI/ML team builds models that researchers and internal tools consume. Different tasks need different models — a fast small model for autocomplete, a large model for complex reasoning, a specialized model for code gen. Instead of each consumer picking a model, the orchestrator decides.

**Features:**
- **Smart routing**: given a request, pick the best model based on task type, complexity, latency requirements, and cost budget
- **Cascading**: sequentially escalate using a calibrated, task-specific quality signal
- **Failover**: retry or switch providers on timeout, overload, or retryable error; this is orthogonal to routing strategy
- **Model registry**: manage versions, A/B tests, canary deployments
- **Unified API**: consumers see one endpoint, orchestrator handles the rest
- **Observability**: per-model metrics, cost tracking, quality monitoring
- **Rate limiting & quotas**: per-team budgets, fair scheduling
- **Request lifecycle**: retry-safe creation, cancellation, durable status, and resumable streaming

---

## 2. Architecture

```
┌──────────────────────────────────────────────────────┐
│                   CONSUMERS                           │
│  (research tools, trading systems, internal apps)     │
│                                                       │
│        All call: POST /v1/responses                   │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│                  API GATEWAY                          │
│  - Auth (API key → team identity)                    │
│  - Rate limiting (per-team quotas)                   │
│  - Request validation                                │
│  - Logging (every request, for audit + billing)      │
└─────────────────────┬────────────────────────────────┘
                      ▼
┌──────────────────────────────────────────────────────┐
│              ROUTING ENGINE (the brain)               │
│                                                       │
│  Input: request + metadata (task_type, urgency,       │
│         budget, quality_requirement)                  │
│                                                       │
│  Decision:                                            │
│    ┌──────────────────────────────┐                  │
│    │  1. Classify task complexity  │                  │
│    │  2. Check routing rules       │                  │
│    │  3. Check model availability  │                  │
│    │  4. Select model + strategy   │                  │
│    │     (direct / cascade / fan)  │                  │
│    └──────────────────────────────┘                  │
│                                                       │
│  Execution strategies:                                │
│    DIRECT   → send to one model                      │
│    CASCADE  → try cheap first, escalate if needed    │
│    FAN-OUT  → send to N models, pick best response   │
│  Error policy: FAILOVER on retryable failure         │
└────────┬──────────┬──────────┬───────────────────────┘
         │          │          │
         ▼          ▼          ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Model A │ │ Model B │ │ Model C │
    │ (7B,    │ │ (70B,   │ │ (code   │
    │  fast)  │ │  smart) │ │  specialist)│
    └────┬────┘ └────┬────┘ └────┬────┘
         │          │          │
         └──────────┼──────────┘
                    ▼
┌──────────────────────────────────────────────────────┐
│              RESPONSE PROCESSOR                       │
│  - Task-specific validation / calibrated risk         │
│  - Cascade decision (good enough or escalate?)        │
│  - Logging (model used, latency, tokens, cost)        │
│  - Return to consumer                                 │
└──────────────────────────────────────────────────────┘
```

The gateway hands each request to a **request coordinator** before routing. It owns the durable state machine, idempotency record, end-to-end deadline, atomic cost reservation, cancellation token, and monotonically sequenced event log. The routing engine receives an immutable registry/policy snapshot and compiles a feasible execution plan.

Keep the control plane (registry updates, experiments, policies) separate from the serving data plane. Serving uses a cached last-known-good snapshot so a registry outage does not stop inference.

---

## 3. Contracts

### 3.1 Unified Consumer API

Consumers don't pick models — they describe what they need.

```jsonc
// POST /v1/responses
// Header: Idempotency-Key: resp-request-7f2
{
    "messages": [
        {"role": "user", "content": "Optimize this pandas groupby for 100M rows"}
    ],
    
    // Routing hints (all optional — orchestrator has sensible defaults)
    "routing": {
        "task_type": "code",              // code | reasoning | chat | extraction | auto
        "max_latency_ms": 3000,           // gateway receipt → terminal event deadline
        "max_ttft_ms": 500,               // time-to-first-token target
        "max_cost_usd": 0.05,             // includes classifier/verifier/judge calls
        "quality": "high",                // routing target, not a correctness guarantee
        "model_override": null            // privileged override; constraints still apply
    },
    
    // Standard generation parameters
    "temperature": 0.2,
    "max_tokens": 2000,
    "stream": true
}
```

**Request lifecycle contract:**

```http
POST   /v1/responses
GET    /v1/responses/{response_id}
DELETE /v1/responses/{response_id}
GET    /v1/responses/{response_id}/events
```

```http
GET /v1/responses/resp_abc123/events
Accept: text/event-stream
Last-Event-ID: 41
```

- Scope `Idempotency-Key` by tenant and endpoint. Same key + same canonical payload returns the original response/execution; same key + different payload returns `409 Conflict`
- `DELETE` is idempotent and propagates cancellation to every active model, verifier, and judge attempt
- SSE event delivery is ordered and at-least-once per response. Events have monotonically increasing IDs; reconnect with `Last-Event-ID`
- If event retention expired, return `410 Gone`; fetch the durable response snapshot/current cursor before reconnecting
- For streaming, commit to one model before emitting the first content delta. Failover is allowed only before that point; never splice one model's prefix with another model's output
- Return `422` when constraints are infeasible, `429` for quota/rate limits, `503` when no healthy compatible route exists, and `504` when the end-to-end deadline expires

### 3.2 Execution Plan (Internal)

```python
class ExecutionPlan:
    strategy: str                   # "direct" | "cascade" | "fan_out"
    models: list[str]               # ordered for direct/cascade
    fallback_on: set[str]           # timeout | overload | retryable_error
    max_attempts: int
    selector: str | None            # fan-out quorum, verifier, or judge
    quality_gate_id: str | None     # versioned task-specific verifier/threshold
    policy_version: str
    registry_version: str
    estimated_max_cost_usd: float
    estimated_p99_latency_ms: float
    deadline_at: datetime
    
class RoutingRule:
    """Configurable rules, not hardcoded."""
    condition: str                  # "task_type == 'code' AND quality == 'high'"
    action: ExecutionPlan
    priority: int                   # higher priority rules evaluated first
```

Failover is an error-handling policy applied to direct, cascade, or fan-out—not a fourth strategy. Admit a plan only if worst-case reserved cost fits request/team budgets and estimated p99 latency fits the remaining deadline. Cascade latency is additive; fan-out latency includes the slowest required branch plus selection.

### 3.3 Model Registry

```python
class ModelEntry:
    model_id: str                   # "tower-7b-fast-v3"
    model_family: str               # "tower-7b"
    version: str                    # "v3"
    status: str                     # "active" | "canary" | "shadow" | "deprecated"
    revision_digest: str            # immutable weights/runtime identity
    api_contract_version: str
    
    # Capabilities
    capabilities: list[str]         # ["code", "reasoning", "chat"]
    context_window: int             # 128000
    supports_streaming: bool
    supports_tools: bool
    supports_structured_output: bool
    
    # Performance profile (measured, not claimed)
    p99_latency_ms: float           # 450
    input_cost_per_1k_tokens: float
    output_cost_per_1k_tokens: float
    quality_scores: dict            # {"code": 0.85, "reasoning": 0.72, "chat": 0.90}
    profile_version: str
    profile_measured_at: datetime
    
    # Serving
    endpoint: str                   # "http://model-7b.internal:8080/v1"
    replicas: int                   # current replica count
    gpu_type: str                   # "A100"
    health_status: str              # "healthy" | "degraded" | "unavailable"
    
    # Traffic
    traffic_weight: float           # 0.05 means 5% of eligible traffic
    canary_of: str | None           # "tower-7b-fast-v2" (if this is a canary)
```

Profiles are segmented by task, input/output size, hardware, and region. A single average latency or quality number is insufficient for deadline admission.

### 3.4 Response (to Consumer)

```jsonc
{
    "id": "resp_abc123",
    "status": "completed",                       // completed | failed | cancelled
    "content": "Here's an optimized approach using chunked processing...",
    "model": "tower-reasoning",                 // stable logical alias
    "routing_metadata": {
        "schema_version": "1",
        "strategy": "cascade",
        "constraint_status": {
            "cost": "met",
            "deadline": "met",
            "quality": "estimated_met"
        }
    },
    "usage": {
        "input_tokens": 150,
        "output_tokens": 800,
        "total_tokens": 950,
        "cost_usd": 0.012
    }
}
```

Concrete deployment IDs, models tried, and escalation reasons belong in an authorized debug view—not the stable consumer contract. Clients ignore unknown additive fields; breaking fields or events require a new API version.

SSE events are sequenced and resumable:

```text
response.created → response.output_delta* →
response.completed | response.failed | response.cancelled
```

---

## 4. Key Design Decisions

### 4.1 The Cascade Pattern (Most Important to Explain)

The biggest cost savings come from NOT using the expensive model for easy requests.

```python
class CascadeExecutor:
    async def execute(self, request, plan: ExecutionPlan):
        for model_id in plan.models:  # ordered cheap → expensive
            self.reserve_cost_and_check_deadline(model_id)
            response = await self.call_model(model_id, request)
            risk = self.task_verifier.assess(response, plan.quality_gate_id)
            
            if risk <= self.quality_gate.threshold(plan.quality_gate_id):
                return response  # good enough, stop here
            
            # Log the escalation for analysis
            self.log_escalation(model_id, risk, request)
        
        raise QualityConstraintNotMet()
```

**Confidence means calibrated risk, not model self-confidence.** Use task-specific evidence: schema checks for extraction, executable tests for code, grounded citation checks for retrieval, or a verifier calibrated on held-out labeled data. Token entropy is often unavailable and incomparable across tokenizers; output length and self-reported confidence are not correctness signals.

If the cheap-model cost is `Ccheap`, expensive-model cost is `Cexpensive`, escalation rate is `p`, and verification cost is `Cverify`, expected cost is:

`Ccheap + p × Cexpensive + Cverify`

A ~60% saving is possible only under specific measured ratios—for example `p = 0.30` and `Ccheap + Cverify ≈ 0.10 × Cexpensive`. Report added tail latency and false accepts alongside cost.

### 4.2 A/B Testing & Canary Deployments

```python
class TrafficSplitter:
    def route_with_experiment(self, request, experiment):
        active_models = self.registry.get_active(experiment.model_family)
        # Stable keyed hash; Python's process-randomized hash() is not suitable.
        key = f"{experiment.id}:{request.authenticated_subject_id}"
        bucket = stable_hmac_bucket(key, buckets=10_000)
        
        cumulative = 0
        for model, traffic_weight in active_models:
            cumulative += int(traffic_weight * 10_000)
            if bucket < cumulative:
                return model
        return active_models[-1]  # fallback
```

Canaries pass contract, safety, capability, and load tests before receiving traffic. Ramp through bounded stages such as 1% → 5% → 10% → 25% → 50% → 100%. Compare against baseline over minimum sample and time windows, stratified by task and request size.

Halt or roll back on safety violations, schema regressions, error rate, p99 latency, cost, or saturation—not just average quality. LLM-as-judge is a versioned proxy and cannot be the sole promotion gate. Keep a kill switch and last-known-good registry snapshot.

### 4.3 Per-Team Quotas and Cost Attribution

```python
class QuotaManager:
    def reserve(self, team_id: str, response_id: str, worst_case_cost: float):
        # One atomic ledger transaction prevents concurrent overspend.
        return self.ledger.compare_and_reserve(
            key=(team_id, response_id),
            amount=worst_case_cost,
        )

    def settle(self, reservation_id: str, actual_cost: float):
        self.ledger.settle_and_release_remainder(reservation_id, actual_cost)
```

Admission reserves worst-case cost for classifiers, models, verifiers, and judges. Every additional dispatch rechecks request budget and deadline. Completion, failure, or cancellation settles actual usage and releases the remainder; a reconciler reclaims expired reservations.

### 4.4 Observability Dashboard

What to monitor (per model, per team, per routing strategy):

```python
metrics = {
    # Quality
    "quality_score": "LLM-as-judge score on sampled responses",
    "cascade_rate": "% of requests that escalated (high = cheap model is weak)",
    "error_rate": "% of requests that failed",
    
    # Cost
    "cost_per_request_avg": "avg $ per request",
    "cost_by_team": "$ spent per team per day",
    "estimated_cost_saved_by_cascade": "counterfactual estimate vs baseline route",
    
    # Latency
    "p50_latency_ms": "median response time",
    "p99_latency_ms": "tail latency",
    "cascade_latency_overhead": "extra latency from failed cheap attempt",
    
    # Throughput
    "requests_per_second": "per model",
    "queue_depth": "per model (are we saturated?)",
    "gpu_utilization": "per model serving instance",
}
```

Also trace every classifier/model/verifier/judge attempt with policy version, registry version, experiment bucket, reserved/actual cost, remaining deadline, and cancellation outcome. Monitor idempotency conflicts, stream resumes/replay gaps, budget rejections, deadline exhaustion, fallback reasons, calibration error, false-accept rate, canary rollbacks, and circuit-breaker trips. Keep high-cardinality team attribution in secured logs/billing storage rather than metrics labels.

---

## 5. Advanced: Fan-Out for Critical Requests

For high-stakes requests (e.g., a trading signal), send to multiple models in parallel, take the best or most-agreed-upon answer.

```python
class FanOutExecutor:
    async def execute(self, request, models: list[str]):
        self.reserve_worst_case_branch_and_selector_cost(models)
        tasks = [
            self.call_model(m, request, timeout=self.remaining_deadline())
            for m in models
        ]
        outcomes = await gather_independent_outcomes(tasks)
        
        # One branch failure does not fail the group.
        best = self.validated_selector.select(outcomes)
        cancel_unfinished_tasks(tasks)
        
        return best
```

Majority vote is valid only for normalized structured outputs such as classifications. Raw confidence is not comparable across models; generative selection needs a calibrated task verifier or separately budgeted/versioned judge.

**Tradeoff**: Fan-out increases cost and may improve robustness only when model errors are sufficiently diverse and selection is validated. It is not a correctness guarantee for high-stakes trading decisions.

---

## 6. Interview Walkthrough (5-min version)

1. **"The API defines retry-safe creation, cancellation, resumable streaming, and explicit deadline/cost semantics."**
2. **"The router compiles a feasible direct, cascade, or fan-out plan from a versioned registry snapshot; failover is orthogonal."**
3. **"Cascade uses task-specific calibrated verification, and every escalation rechecks remaining deadline and reserved cost."**
4. **"Streaming commits before the first content delta, so outputs from different models are never spliced."**
5. **"Canaries use stable assignment, staged ramps, multidimensional guardrails, and a kill switch."**
6. **"Atomic reservations, per-attempt traces, calibration metrics, and durable request state make the system auditable."**
