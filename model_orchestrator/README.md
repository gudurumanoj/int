# Standalone Model Orchestrator Demo

A small Python 3.11/FastAPI implementation of `design_model_orchestrator.md`.
It demonstrates routing, durable request state, retry-safe creation, cancellation,
cost accounting, and resumable events without hiding the control flow behind a
framework.

The included tests use injected fake upstreams. Running the tests never calls a
network or an LLM. The sample model URLs use the reserved `.invalid` domain.

## Setup and run

```powershell
cd C:\Users\manoj.guduru\Downloads\int_prep\model_orchestrator
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
```

Set the per-target API key environment variables named in `models.json` only if
you intentionally want to call real OpenAI-compatible services. Keys belong in
the environment, never in `models.json`.

```powershell
uvicorn app.main:create_app --factory --reload --port 8003
```

Run the test suite:

```powershell
python -m pytest
```

## Configuration

`.env.example` documents the `ORCH_` application settings:

- `SQLITE_PATH` and `MODEL_REGISTRY_PATH`
- idempotency TTL and event retention in seconds
- default end-to-end timeout and retry count
- default maximum output tokens and temperature
- `DEFAULT_API_KEY_ENV`, conventionally `OPENAI_API_KEY`

Every target in `models.json` has a logical `id`, OpenAI-compatible `base_url`,
`api_key_env`, provider-specific `upstream_model`, capabilities, quality tier,
p99 latency, input/output prices per 1,000 tokens, health, and an optional
quality gate. The per-model `api_key_env` is authoritative; the application
default is the convention for registries that derive entries programmatically.

`models.json` is loaded as an immutable serving snapshot at startup. Only
healthy targets with the requested capability are eligible.

## Request coordination walkthrough

`Orchestrator.create` is the coordinator entry point:

1. Resolve token and temperature defaults, then serialize one canonical request
   payload for idempotency comparison.
2. Compile a route from the immutable registry snapshot. Capability and health
   filtering happen before cost/deadline admission, so rejected requests are
   never dispatched.
3. In one SQLite transaction, claim the tenant/endpoint idempotency key, create
   the response snapshot, reserve worst-case cost, and append event 1
   (`response.created`).
4. Start one execution context containing the absolute deadline, a shared
   `asyncio.Event`, and the task handles for the coordinator, fan-out branches,
   cancellation waiter, and current upstream call.
5. Persist each attempt and its usage. Responses stay private to their branch
   until a strategy selects one complete result.
6. Commit only the selected content, append the terminal events, calculate
   aggregate actual cost, and settle the reservation. Failure and cancellation
   use the same terminal settlement path.

The POST returns the durable snapshot while execution continues. GET reads that
snapshot directly from SQLite; DELETE signals and cancels live work before
persisting the idempotent terminal state; the SSE endpoint replays the durable
event log and then follows it until a terminal event.

## Routing semantics

Consumers specify needs, not provider model names. `routing.strategy` can be:

- `direct`: select the closest configured quality tier. Retryable timeouts,
  connection failures, overload, and 5xx failures can retry or move to another
  compatible target. A successful non-retryable result does not silently
  fail over.
- `cascade`: for JSON requests only, try gated targets from inexpensive to more
  capable and stop at the first output accepted by deterministic `json.loads`
  validation.
- `fan_out`: for JSON requests only, run independent branches and select the
  first valid JSON result. Once selected, unfinished branches are cancelled.

Failover is an error policy shared by the strategies, not a fourth strategy.
`auto` and non-JSON cascade/fan-out requests route directly to the configured
quality tier. The app never treats model self-confidence or output length as a
calibrated correctness score.

Before creating durable work, the router reserves a conservative maximum cost
using estimated input tokens, requested maximum output tokens, all planned
targets, and retries. It similarly checks additive p99 latency for sequential
plans and maximum branch p99 for fan-out. Infeasible request budgets or
deadlines return `422`; no healthy compatible route returns `503`.

Every upstream result is buffered. Only the selected result becomes a content
event, so output from different models is never spliced. Actual cost is computed
from reported input/output usage. Completion, failure, and cancellation settle
the SQLite reservation and release the unused estimate.

## API

Create work with an idempotency key scoped by tenant and endpoint:

```bash
curl -i http://127.0.0.1:8003/v1/responses \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: demo-001" \
  -H "X-Tenant-ID: interview-demo" \
  -d '{
    "messages": [{"role": "user", "content": "Return a JSON greeting"}],
    "response_format": "json",
    "routing": {
      "task_type": "extraction",
      "strategy": "cascade",
      "quality": "high",
      "max_latency_ms": 5000,
      "max_cost_usd": 0.05
    }
  }'
```

Creation returns `202` with a durable `in_progress` snapshot. Reusing the same
key and canonical payload returns the original response with
`Idempotent-Replay: true`; changing the payload returns `409`.

```bash
curl http://127.0.0.1:8003/v1/responses/resp_ID
curl -X DELETE http://127.0.0.1:8003/v1/responses/resp_ID
curl -N http://127.0.0.1:8003/v1/responses/resp_ID/events \
  -H "Accept: text/event-stream" \
  -H "Last-Event-ID: 1"
```

DELETE is idempotent. It sets the shared cancellation event, cancels active task
handles, persists `cancelled`, and settles cost. Events are stored with a
per-response monotonically increasing ID and replayed after `Last-Event-ID`.
Requests older than retained history return `410`.

## Durable data

SQLite stores response snapshots, tenant-scoped idempotency records, ordered
events, every upstream attempt, and cost reservations. Response creation,
idempotency claim, initial event, and reservation are committed together.

## Test examples

`tests/test_orchestrator.py` injects an in-memory fake at the `Upstream`
protocol boundary. Its examples cover:

- same-key replay and different-payload conflict
- direct quality-tier selection and JSON cascade escalation
- retryable direct failure moving to a fallback without changing strategy
- budget and p99 deadline rejection before any fake dispatch
- idempotent DELETE and cancellation reaching a blocked fake call
- fan-out selection and cancellation of unfinished branches
- SSE replay starting strictly after `Last-Event-ID`

No test requires an API key, reachable model URL, or network access.

## Educational limitations

- This is a single-process demo with synchronous SQLite calls and in-memory
  active task handles. It does not resume unfinished work after process restart.
- Tenant identity is a header demonstration, not authentication. There is no
  team quota ledger, rate limiter, or cross-request budget transaction.
- Token admission uses a simple character estimate rather than a
  provider-specific tokenizer. Usage and pricing are trusted as configured.
- JSON syntax is the only concrete quality gate; schema validation, code tests,
  grounded citation checks, and calibrated verifiers are intentionally absent.
- Fan-out uses first-valid selection, not majority voting or an LLM judge.
- SSE uses durable polling and emits the selected buffered content once. The
  accepted `stream` field does not enable upstream token streaming.
- Registry refresh, canaries, metrics, tracing, tool calls, and structured
  provider error normalization are outside this minimal demo.
