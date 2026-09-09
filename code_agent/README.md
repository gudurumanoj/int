# Educational Code Agent

A compact Python 3.11/FastAPI implementation of the session/run code-agent
contracts in `design_code_agent.md`. It demonstrates durable state,
idempotency, structured tool calls, risk-based approval, cancellation, and
resumable server-sent events without hiding the mechanics behind a large
framework.

This is educational code, **not a hardened production sandbox**. Docker with
resource flags is not a complete security boundary. A production system also
needs rootless isolation or microVMs, syscall policies, disk quotas, stronger
symlink/race defenses, authentication and tenant-scoped idempotency, secret
brokering/redaction, audit retention, and crash reconciliation.

## Architecture and flow

- `app/api.py` defines the FastAPI lifespan and HTTP/SSE contracts. New runs
  are held by a small in-process background-task set and cancelled on shutdown.
- `app/storage.py` owns SQLite transactions. It stores sessions, runs,
  idempotency records, per-run monotonic events, approvals, and tool-call
  intents/results.
- `app/llm.py` is a thin adapter over
  `AsyncOpenAI(base_url=..., api_key=...)`. Tests supply a fake implementing
  the same small `ChatClient` protocol.
- `app/runtime.py` runs one LLM loop: request a structured turn, validate tool
  calls, apply deterministic policy, execute each call once, persist the
  observation, and continue. Only public summaries and structured records are
  stored; chain-of-thought is never parsed or persisted.
- `app/tools.py` confines file reads/writes to `workspaces/<run_id>`. Writes
  use an atomic replace. `shell` requires approval and invokes a dedicated
  networkless Docker container with CPU, memory, PID, temporary-filesystem, and
  wall-time limits. Cancellation removes the container and terminates the
  Docker client process tree.

The normal flow is:

1. Create a durable session with an `Idempotency-Key`.
2. Create a bounded run; the API starts one understandable background task.
3. The LLM returns public text and structured `tool_calls`.
4. The runtime validates and records each call before any side effect.
5. `shell` pauses the run and emits `approval.required`; file tools run in the
   per-run workspace.
6. Events are committed before SSE delivery. Reconnect with `Last-Event-ID` to
   replay strictly newer sequences.
7. Completion, failure, or cancellation commits one terminal run state.

The same scoped idempotency key and canonical request returns the original
response. Reusing the key with a different request returns HTTP 409. Reused
tool-call IDs return the committed result; an unfinished call fails closed
because its side-effect outcome may be ambiguous.

### Important invariants

- Session/run creation and its idempotency record commit in one SQLite
  transaction.
- Event sequence allocation and insertion share the same transaction and
  process lock, so each run has a monotonic replay cursor.
- Approval is decided from the stored proposal hash, expiry, and current run
  state. The model cannot mark its own command safe.
- Cancellation first moves the run to `cancelling`, revokes pending approvals,
  stops the active runner, and then compare-and-sets `cancelled`.
- A tool intent is durable before dispatch. A committed result is replayed;
  an unfinished intent is not blindly executed again.

## Configuration

Copy `.env.example` to `.env` and set:

- `OPENAI_BASE_URL` — OpenAI-compatible API root; default
  `https://api.openai.com/v1`.
- `OPENAI_API_KEY` — API credential; required unless a fake client is
  injected.
- `OPENAI_MODEL` — default Chat Completions model.
- `OPENAI_TIMEOUT_SECONDS` — provider request timeout.
- `OPENAI_MAX_RETRIES` — retries performed by the OpenAI SDK.
- `OPENAI_TEMPERATURE` — sampling temperature.
- `OPENAI_MAX_TOKENS` — maximum generated tokens per model turn.
- `CODE_AGENT_DB_PATH` — SQLite file; default `./code_agent.db`.
- `CODE_AGENT_WORKSPACE_ROOT` — parent of isolated per-run workspaces; default
  `./workspaces`.
- `CODE_AGENT_DOCKER_IMAGE` — shell image; default `python:3.11-slim`.
- `CODE_AGENT_APPROVAL_TTL_SECONDS` — approval lifetime; default 300.
- `CODE_AGENT_SSE_HEARTBEAT_SECONDS` — idle SSE heartbeat interval; default 15.

## Run locally

Docker must be installed only if an approved `shell` tool call will run. The
service and tests themselves do not start Docker.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
# Edit .env, especially OPENAI_API_KEY.
uvicorn app.api:app --reload --env-file .env --port 8001
```

Run tests with:

```powershell
pytest
```

Tests use temporary SQLite databases, temporary workspaces, fake LLM/runner
objects, and never make network or Docker calls.

The focused examples are organized by responsibility:

- `tests/test_idempotency.py` covers replay and payload conflicts.
- `tests/test_events.py` covers replay strictly after a sequence.
- `tests/test_approvals.py` covers argument-hash mismatch, repeated decisions,
  conflicting decisions, and stale/revoked approvals.
- `tests/test_runtime.py` demonstrates fake-client injection, normal
  completion, and idempotent cancellation without invoking Docker.

## Endpoint examples

Create a session:

```bash
curl -i -X POST http://127.0.0.1:8001/v1/agent-sessions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: create-session-demo" \
  -d '{"workspace_id":"workspace-demo"}'
```

Create a run:

```bash
curl -i -X POST \
  http://127.0.0.1:8001/v1/agent-sessions/sess_REPLACE/runs \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: run-demo-1" \
  -d '{"input":"Create hello.py and test it","base_workspace_revision":"initial","config":{"max_steps":10,"max_wall_time_seconds":120,"approval_policy":"risk_based"}}'
```

Read status, cancel safely, or reconnect to events:

```bash
curl http://127.0.0.1:8001/v1/agent-runs/run_REPLACE

curl -X POST http://127.0.0.1:8001/v1/agent-runs/run_REPLACE/cancel \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: cancel-run-demo" \
  -d '{"reason":"User requested cancellation"}'

curl -N http://127.0.0.1:8001/v1/agent-runs/run_REPLACE/events \
  -H "Accept: text/event-stream" \
  -H "Last-Event-ID: 4"
```

An `approval.required` event contains the approval ID and exact
`arguments_hash`. Decide it once:

```bash
curl -X PUT \
  http://127.0.0.1:8001/v1/approval-requests/apr_REPLACE/decision \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: decide-approval-demo" \
  -d '{"decision":"allow_once","expected_arguments_hash":"sha256:REPLACE","comment":"Reviewed command"}'
```
