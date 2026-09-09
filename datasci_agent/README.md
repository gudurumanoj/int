# Standalone data-science agent demo

This is a small Python 3.11/FastAPI teaching implementation of the
planner-worker design in `../design_datasci_agent.md`. It keeps orchestration
explicit: the **planner** LLM returns a strict, versioned step DAG; the
**worker** LLM proposes code for one runnable step; ordinary application code
profiles data, validates policy, requests approval, invokes the sandbox, and
atomically commits results.

The planner and worker are separate Chat Completions calls. By default they use
the same model and the same shared `AsyncOpenAI(base_url=..., api_key=...)`
client. They can be different logical roles without being different models.
The orchestrator is deterministic Python code, not another LLM.

## Flow

1. Register CSV text. The service hashes the exact UTF-8 bytes and stores an
   immutable, hash-addressed snapshot. Changed content under the same filename
   creates the next dataset version.
2. Create a durable session and submit a run with an idempotency key.
3. A deterministic bounded CSV profile is persisted as an event before
   planning.
4. The planner returns strict JSON containing a topologically ordered step DAG.
5. For each step, the worker proposes code. AST/policy checks run before any
   sandbox call.
6. `target_change`, `row_drop`, and `package_install` risk tags always require
   approval. The decision is bound to the exact proposal hash and plan version.
7. The illustrative runner invokes Docker with no network, CPU/memory/PID/time
   limits, a read-only dataset mount, a read-only code workspace, and a scoped
   writable output mount.
8. A successful step commits artifact rows, a result, a checkpoint manifest,
   and a monotonic event in one SQLite transaction.
9. The final Markdown report links evidence by path/hash. A reproducibility
   manifest records dataset hashes, sandbox image, dependency file, model and
   prompt refs, plan hash, seed, and code/artifact hashes.

Runs do not depend on a persistent Python kernel. SQLite and committed files
are the source of truth, so an approved run can resume from its last checkpoint.

## Setup and run

```powershell
cd C:\Users\manoj.guduru\Downloads\int_prep\datasci_agent
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[test]"
Copy-Item .env.example .env
# Export the values from .env using your preferred environment loader.
uvicorn app.main:app --reload --port 8002
```

The service does not load `.env` itself. This keeps configuration behavior
obvious; set the variables in the shell or use an external dotenv launcher.

Run tests:

```powershell
python -m pytest
```

Tests inject a fake planner/worker and a fake runner. They make no network/LLM
calls and never invoke Docker.

## Configuration

LLM settings:

- `OPENAI_BASE_URL`
- `OPENAI_API_KEY`
- `OPENAI_MODEL`
- `OPENAI_TIMEOUT_SECONDS`
- `OPENAI_MAX_RETRIES`
- `OPENAI_TEMPERATURE`
- `OPENAI_MAX_TOKENS`

Service/sandbox settings:

- `DATASCI_DATA_DIR` (default `./data`)
- `DATASCI_DOCKER_IMAGE` (default `datasci-sandbox:py311`)
- `DATASCI_DOCKER_CPUS`
- `DATASCI_DOCKER_MEMORY`
- `DATASCI_DOCKER_TIMEOUT_SECONDS`

The Docker image is intentionally not built or supplied by this demo. In a
real deployment, use a pinned digest and a reviewed dependency lock.

## API

- `POST /v1/datasci/datasets` — register immutable CSV JSON
- `POST /v1/datasci/sessions` — create a session
- `POST /v1/datasci/sessions/{session_id}/runs` — submit a run
- `GET /v1/datasci/runs/{run_id}` — status and artifact references
- `POST /v1/datasci/runs/{run_id}/cancel` — idempotent cancellation
- `GET /v1/datasci/runs/{run_id}/events` — resumable SSE
- `PUT /v1/datasci/approval-requests/{approval_id}/decision` — exact proposal
  decision

The session, run, cancellation, and approval mutations use
`Idempotency-Key`. Reusing a key with the same canonical JSON replays the
stored response; changing the payload returns `409 Conflict`.

Example:

```bash
curl -X POST http://127.0.0.1:8002/v1/datasci/datasets \
  -H "Content-Type: application/json" \
  -d '{"filename":"signals.csv","content":"feature,target\n1,0\n2,1\n"}'

curl -X POST http://127.0.0.1:8002/v1/datasci/sessions \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: signal-session-v1" \
  -d '{"title":"Signal analysis"}'

curl -X POST http://127.0.0.1:8002/v1/datasci/sessions/SESSION_ID/runs \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: signal-run-v1" \
  -d '{
    "question":"Which features predict the target?",
    "datasets":[{"dataset_id":"DATASET_ID","version":1,"sha256":"SHA256"}],
    "config":{
      "target_column":"target",
      "time_column":null,
      "group_column":null,
      "random_seed":42
    }
  }'

curl -N http://127.0.0.1:8002/v1/datasci/runs/RUN_ID/events \
  -H "Accept: text/event-stream" \
  -H "Last-Event-ID: 12"

curl -X PUT \
  http://127.0.0.1:8002/v1/datasci/approval-requests/APPROVAL_ID/decision \
  -H "Content-Type: application/json" \
  -H "Idempotency-Key: approve-proposal-v1" \
  -d '{
    "decision":"approve",
    "expected_proposal_hash":"sha256:PROPOSAL_HASH",
    "expected_plan_version":1,
    "comment":"Reviewed"
  }'
```

Approval IDs and proposal hashes appear in the `approval.required` SSE event.
SSE event IDs are monotonically increasing per run; reconnect with
`Last-Event-ID` and deduplicate because delivery is at least once.

## Statistical-validity guardrails

The profiler and report make the following checklist visible:

- choose chronological/walk-forward splits for temporal data;
- keep repeated groups in only one split;
- split before preprocessing and fit transforms only on training folds;
- keep a final holdout untouched during model selection;
- freeze and verify the target, and warn about target-derived/post-outcome
  leakage.

These are educational checks and prompts, not proof that arbitrary generated
code is statistically valid.

## Limitations

- This is not AutoML and does not claim causal identification.
- CSV profiling is intentionally small and uses only the standard library.
- The demo has a single-process scheduler and local SQLite/filesystem storage;
  production needs a durable queue, tenant isolation, retention, quotas, and
  object storage.
- Output formats are restricted to textual allowlisted files. Chart rendering
  and model serialization are left as extensions.
- Package-install proposals require approval, but the sandbox has no network
  and a read-only root, so installation will normally fail. Production should
  use reviewed, prebuilt images.
- Cancellation marks durable state first and then terminates the tracked
  process; production should also reconcile orphaned containers after crashes.
- External LLM output and some numerical kernels are not bit-for-bit
  reproducible even when the recorded seed and inputs are identical.
