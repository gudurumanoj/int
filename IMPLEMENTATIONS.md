# Minimal AI System Implementations

These four directories turn the design notes into small, readable Python examples. They are intentionally separate so each can be studied as if it were its own service.

| Design | Example | Main idea |
|---|---|---|
| [Autonomous code agent](design_code_agent.md) | [code_agent](code_agent/README.md) | One LLM proposes tool calls; deterministic code owns policy, state, approvals, and execution |
| [Data-science agent](design_datasci_agent.md) | [datasci_agent](datasci_agent/README.md) | Planner and worker LLM calls operate inside a versioned, checkpointed workflow |
| [Model orchestrator](design_model_orchestrator.md) | [model_orchestrator](model_orchestrator/README.md) | A request coordinator selects direct, cascade, or fan-out execution across compatible endpoints |
| [LLM-assisted recommendations](design_llm_recsys.md) | [llm_recsys](llm_recsys/README.md) | Deterministic retrieval/ranking remains primary; an LLM optionally reranks and explains |

## Common OpenAI-Compatible Configuration

Each LLM-using service reads the following settings:

```env
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_API_KEY=replace-me
OPENAI_MODEL=replace-me
OPENAI_TIMEOUT_SECONDS=60
OPENAI_MAX_RETRIES=2
OPENAI_TEMPERATURE=0.2
OPENAI_MAX_TOKENS=2000
```

`OPENAI_BASE_URL` can point at OpenAI or another server that implements the OpenAI Chat Completions contract. The code uses structured outputs or tool calls where the design needs them. It never parses private chain-of-thought.

The model orchestrator is different because it routes among several upstreams. Its `models.json` contains one base URL, model name, API-key environment-variable name, capability profile, cost profile, and latency profile per target.

## How to Read the Examples

Read each directory in this order:

1. `README.md` — request flow and design boundaries.
2. `app/schemas.py` — public and internal contracts.
3. `app/main.py` — HTTP endpoints only.
4. Storage/repository module — idempotency, states, events, and atomic decisions.
5. Service/coordinator module — application workflow.
6. LLM and execution adapters — external side effects behind interfaces.
7. Tests — executable examples of the important guarantees.

The recurring dependency direction is:

```text
HTTP endpoint
    → application service / coordinator
        → durable repository
        → policy and validation
        → LLM or execution adapter
```

The LLM is a decision-making dependency. It does not own authorization, budgets, idempotency, state transitions, cancellation, or audit history.

## The Four Operational Contracts

### Idempotency

The server scopes an `Idempotency-Key` to a caller and operation, hashes the canonical request body, and stores the resulting resource ID.

- Same key and body: return the first result.
- Same key and different body: return `409 Conflict`.
- No key: reject operations where retry safety is required.

This prevents a client retry from creating another run, spending twice, or recording duplicate feedback.

### Cancellation

Cancellation is a durable state transition, not just an in-memory task cancellation:

```text
queued | running | waiting_for_approval
    → cancelling
    → cancelled
```

The coordinator stops new work, signals active adapters, rejects late commits, and atomically races completion so only one terminal state wins.

### Approval Decisions

An approval request is bound to the exact proposed action:

```text
approval_id + run_id + proposal_hash + expiry + policy_version
```

If an LLM changes the tool arguments, the old approval cannot authorize the new action. Repeated identical decisions are safe; conflicting or stale decisions return `409`.

### Event Resumption

Events are stored before they are sent. Each run has a monotonic sequence:

```text
1 run.queued
2 run.started
3 tool.started
4 approval.required
```

SSE clients reconnect using `Last-Event-ID`. The server replays events after that sequence. Delivery is at-least-once, so clients deduplicate by event ID.

## Deliberate Simplifications

- SQLite stands in for a transactional database and durable queue.
- FastAPI background tasks stand in for distributed workers.
- Docker commands illustrate a sandbox boundary but are not a hardened multi-tenant runtime.
- Cost and latency profiles are configuration, not live benchmark data.
- Recommendation retrieval uses a small deterministic baseline rather than a production ANN index and learned ranker.
- Tests use fake LLM/execution adapters; understanding the contracts does not require API credentials or Docker.

No dependencies, servers, Docker containers, tests, or live LLM calls need to be run to study the code.
