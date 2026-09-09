# LLM-assisted recommendation service demo

A small Python 3.11/FastAPI service that demonstrates the online path from the
design: static catalog retrieval, eligibility filtering, deterministic
token-overlap ranking, and an optional LLM rerank/explanation over a small
shortlist. It intentionally exposes no probability-like relevance scores.

The service is deterministic without an API key. Tests always inject a fake
reranker and never make network or LLM calls.

## Request flow

1. Resolve the tenant from `X-Tenant-ID` (or `DEFAULT_TENANT_ID`).
2. Check the tenant-scoped `Idempotency-Key` and canonical request hash.
3. Read `app/data/items.json`; filter by tenant, availability, placement,
   locale, and the optional category allowlist.
4. Rank eligible items by deterministic token overlap with placement, query,
   and explicitly supplied interests. Item ID breaks ties.
5. If `explain=true` and an API key configured a reranker, send only the small
   shortlist to an OpenAI-compatible Chat Completions endpoint.
6. Enforce a strict JSON schema, contiguous ranks, unique IDs, and the
   shortlist allowlist. A timeout or invalid response falls back to the
   deterministic order and marks the response degraded.
7. Return grounded reasons and opaque HMAC-signed tracking tokens. Store the
   recommendation set, items, and idempotent response in SQLite.

LLM reasons are used only when they overlap both supplied preference terms and
catalog evidence; otherwise the server emits a deterministic catalog-grounded
reason. Prompt instructions are defense in depth, not a security boundary.
No tools are supplied to the model.

## Local setup

```bash
cd llm_recsys
python -m venv .venv
# Windows PowerShell:
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
Copy-Item .env.example .env
python -m uvicorn app.main:app --reload --port 8004
```

Leave `OPENAI_API_KEY` empty for the fully local deterministic route. SQLite
creates `recsys.db` on first startup.

Run tests:

```bash
python -m pytest
```

## Endpoints

`POST /v1/recommendations`

- Requires `Idempotency-Key`.
- Accepts `request_id`, `user_id`, `session_id`, `placement`, `context`,
  `num_results`, and `explain`.
- The optional context fields are `locale`, `device`, `query`, and `interests`.
- Same tenant, key, and canonical payload replays the exact response. Reusing
  the key with a different payload returns `409`.

```bash
curl -X POST http://127.0.0.1:8004/v1/recommendations \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: req_7f" \
  -H "X-Tenant-ID: demo" \
  -d '{"request_id":"req_7f","user_id":"u_12345","session_id":"sess_456","placement":"homepage","context":{"locale":"en-US","device":"desktop","interests":["volatility","transformers"]},"num_results":3,"explain":true}'
```

`POST /v1/recommendations/{set_id}/feedback`

- Requires `Idempotency-Key` and an immutable `event_id`.
- Accepts `click`, `dismiss`, `bookmark`, or `dwell_60s`.
- Verifies that the token binds tenant, set, item, rank position, user,
  session, and server-controlled expiry.
- An exact same-key retry returns the original response. The same event under
  another key is deduplicated. Changed key payloads or changed event payloads
  return `409`.

Use the set ID and tracking token returned above:

```bash
curl -X POST http://127.0.0.1:8004/v1/recommendations/rec_REPLACE/feedback \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: evt_98" \
  -H "X-Tenant-ID: demo" \
  -d '{"event_id":"evt_98","tracking_token":"REPLACE","item_id":"paper_regime_transformers","action":"click","position":1,"session_id":"sess_456","client_timestamp":"2026-09-08T12:00:00Z","dwell_ms":null}'
```

Interactive API docs are at `http://127.0.0.1:8004/docs`; health is at
`GET /health`.

## Configuration

OpenAI-compatible client settings:

- `OPENAI_BASE_URL`, `OPENAI_API_KEY`, and `OPENAI_MODEL`
- `OPENAI_TIMEOUT_SECONDS` and `OPENAI_MAX_RETRIES`
- `OPENAI_TEMPERATURE` and `OPENAI_MAX_TOKENS`

Storage and security:

- `DATABASE_PATH`, `DEFAULT_TENANT_ID`, and `HMAC_SECRET`
- `RECOMMENDATION_TTL_SECONDS` and `IDEMPOTENCY_TTL_SECONDS`

Routing and eligibility:

- `LLM_SHORTLIST_SIZE`
- `ELIGIBILITY_MAX_CANDIDATES`
- `ELIGIBILITY_REQUIRE_AVAILABLE`
- `ELIGIBILITY_ALLOWED_CATEGORIES` as an optional comma-separated allowlist
- `CATALOG_PATH` if a different static catalog is needed

Replace the development HMAC secret before any non-local deployment.

## Scope and production limitations

This is an explicit educational demo, not a production recommender. It uses a
single-process lock around writes, static request-time interests instead of
consented server-side profiles, lexical ranking instead of ANN retrieval or a
learned ranker, and SQLite instead of a horizontally scalable data store. It
does not implement rate limits, a circuit breaker, encryption, deletion and
retention workflows, ACL/policy services, observability, model/prompt
versioning, impression propensity logging, randomized holdouts, or multi-region
consistency. Cross-process idempotency reservation would need stronger
transaction coordination.

Minimize user data sent to a model, define vendor training/residency/logging
policies, avoid sensitive inferred traits, and support access/export/deletion.
Reasons are predictions constrained to supplied evidence; they are not proof
of user preference.

Feedback is biased by exposure, position, item quality, and whether an
explanation was shown. Clicks are not unbiased labels. Production evaluation
should log impressions and treatment propensity, use randomized holdouts, and
consider IPS or doubly robust analysis. Protect the event path against bots,
poisoning, popularity reinforcement, and echo chambers.

Recommendation serving here is short-lived and synchronous, so it deliberately
has no durable cancel, approval, or resumable SSE state machine. Long-running
batch generation should instead use an asynchronous agent run pattern: create
a durable run, expose status/cancel, persist checkpoints, and resume an event
stream from a cursor.
