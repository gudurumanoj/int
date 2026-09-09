# Design: Autonomous Code Agent (Codex / Claude Code)

## 1. Problem Framing — Think Out Loud

> "Design an autonomous coding agent that can take a natural language task, plan an approach, write code, execute it, observe results, and iterate until the task is done."

The contracts below define a provider-neutral service you could build. They are not the exact public API of Cursor, Codex, or Claude Code, although those products expose similar agent/thread/run concepts.

**First question to ask yourself (or the interviewer):** Who is the user and what's the scope?

- **User**: Software engineer at Tower who wants to automate repetitive coding tasks — data pipeline scripts, model evaluation boilerplate, bug fixes
- **Scope**: Works in a sandboxed environment, has access to a codebase, terminal, and file system
- **Key constraint**: Must be safe (can't `rm -rf /`), cost-bounded, and observable

**Features to call out:**

- Multi-turn execution: agent plans → writes code → runs → reads output → fixes → repeats
- Tool use: file read/write, terminal execution, web search for docs
- Memory: conversation history + working memory of what files it has seen/edited
- Safety: sandboxed execution, cost/step limits, human-in-the-loop checkpoints
- Durability: a session keeps conversation/workspace state; each prompt creates a bounded, cancellable run, progress streams can reconnect, and follow-ups create new runs

---



## 2. High-Level Architecture

```
┌─────────────────────────────────────────────────┐
│                   USER / CLI                     │
│            (natural language task)                │
└──────────────────┬──────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────┐
│              ORCHESTRATOR (Agent Loop)            │
│                                                  │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐    │
│  │  PLANNER │──▶│ EXECUTOR │──▶│ OBSERVER │    │
│  │(reason   │   │(pick tool│   │(read     │    │
│  │ + plan)  │   │+ call it)│   │ output)  │────┤
│  └──────────┘   └──────────┘   └──────────┘    │
│       ▲                                  │       │
│       └──────────────────────────────────┘       │
│                 (loop until done)                 │
│                                                  │
│  ┌────────────────────────────────────────────┐  │
│  │          CONTEXT MANAGER                   │  │
│  │  - conversation history (truncated)        │  │
│  │  - working memory (files seen, plan state) │  │
│  │  - system prompt + tool definitions        │  │
│  └────────────────────────────────────────────┘  │
└──────────────────┬──────────────────────────────┘
                   │
        ┌──────────┼──────────┐
        ▼          ▼          ▼
   ┌─────────┐ ┌────────┐ ┌─────────┐
   │ SANDBOX │ │ FILE   │ │ SEARCH  │
   │(docker  │ │ SYSTEM │ │(docs,   │
   │ exec)   │ │(R/W)   │ │ web)    │
   └─────────┘ └────────┘ └─────────┘
```

**These boxes are responsibilities, not necessarily separate services or models:**

- **Planner**: normally the primary LLM. It proposes a plan, a structured tool call, or completion.
- **Executor/orchestrator**: ordinary application code. It validates schemas, enforces policy and budgets, obtains approval, dispatches tools, and persists state.
- **Observer**: a deterministic adapter that turns tool output into a bounded, structured observation for the next LLM turn.
- A separate planner, critic, or verifier LLM is optional. The basic loop needs only one LLM.

**The core loop (ReAct pattern):**

```python
while run.status == "running":
    check_cancellation()
    reserve_step_budget()

    # Modern model APIs return structured tool calls; do not parse <think> text.
    response = llm.generate(context_projection, tool_schemas)
    persist_public_summary(response.summary)

    if not response.tool_calls:
        validate_acceptance_gates()
        complete_run(response.final_answer)
        break

    for call in response.tool_calls:
        validate_schema_and_policy(call)
        obtain_approval_if_required(call)
        result = execute_tool_once(call, idempotency_key=call.id)
        persist_tool_result(call.id, result)

    checkpoint_run()
```

Private chain-of-thought is not stored or streamed. The durable transcript contains user-visible summaries, structured tool calls, results, approvals, and state transitions.

---



## 3. Contracts / API Design



### 3.1 Durable Session and Run Creation

A **session** owns conversation/workspace state. A **run** is one execution triggered by a user turn. Continuing a completed session creates a new run; it does not reopen the old run.

```http
POST /v1/agent-sessions
Idempotency-Key: create-session-7f2

{"workspace_id": "ws_abc123"}

201 Created
{"session_id": "sess_xyz789", "status": "idle"}
```

```http
POST /v1/agent-sessions/sess_xyz789/runs
Idempotency-Key: run-vwap-001
Content-Type: application/json

{
  "input": "Write a Python script that reads trades.csv, computes VWAP per symbol, and saves to vwap_output.csv",
  "base_workspace_revision": "rev_42",
  "config": {
    "max_steps": 50,
    "max_input_tokens": 100000,
    "max_output_tokens": 20000,
    "max_cost_usd": 2.0,
    "max_wall_time_seconds": 300,
    "model": "configured-default",
    "approval_policy": "risk_based"
  }
}

202 Accepted
{
  "run_id": "run_456",
  "status": "queued",
  "status_url": "/v1/agent-runs/run_456",
  "events_url": "/v1/agent-runs/run_456/events"
}
```

**Idempotency contract:**

- Scope keys by authenticated caller and endpoint, and retain them for a documented TTL.
- Same key + same canonical payload returns the original resource and does not repeat LLM/tool work.
- Same key + different payload returns `409 Conflict`.
- Concurrent runs against one mutable workspace are rejected, queued, or isolated on separate revisions.

### 3.2 Run Status and Cancellation

```text
queued → running ↔ waiting_for_approval
queued | running | waiting_for_approval → cancelling → cancelled
queued | running | waiting_for_approval → failed | budget_exhausted
running → completed
```

```http
GET /v1/agent-runs/run_456

POST /v1/agent-runs/run_456/cancel
Idempotency-Key: cancel-run-456
{"reason": "User requested cancellation"}
```

Cancellation stops scheduling new calls, revokes pending approvals, terminates the active process tree gracefully, then force-kills it after a deadline. Repeated cancellation is safe. A compare-and-set terminal transition resolves a cancel/complete race; exactly one terminal status wins.

### 3.3 Approval Request and Decision

Approval is determined by runtime policy using the tool, arguments, user, workspace, and risk—not by a model-controlled boolean.

```python
class ApprovalRequest:
    approval_id: str
    run_id: str
    tool_call_id: str
    arguments_hash: str
    workspace_revision: str
    risk_summary: str
    expires_at: datetime
    policy_version: str
```

```http
PUT /v1/approval-requests/apr_123/decision
Idempotency-Key: decide-apr-123

{
  "decision": "allow_once",
  "expected_arguments_hash": "sha256:...",
  "comment": "Approved package installation"
}
```

Supported decisions are `allow_once` and `deny`. A repeated identical decision returns the stored result; a conflicting, stale, or expired decision returns `409 Conflict`. If tool arguments change, the runtime creates a new approval request.

### 3.4 Agent → LLM and Tool Contracts

```python
# Provider-neutral structured model response
{
    "summary": "I will inspect the CSV headers before writing the calculation.",
    "tool_calls": [{
        "id": "call_17",
        "name": "read_file",
        "arguments": {"path": "trades.csv"}
    }],
    "finish_reason": "tool_calls"
}
```

```python
class Tool:
    name: str
    description: str
    parameters: dict              # strict JSON Schema
    timeout_seconds: int

class ToolResult:
    tool_call_id: str
    tool_name: str
    status: str                   # succeeded | failed | cancelled | timed_out
    stdout: str                   # bounded; large output becomes an artifact
    stderr: str
    exit_code: int | None
    error_code: str | None
    retryable: bool
    side_effect_summary: dict
    started_at: datetime
    finished_at: datetime
```

Persist tool-call intent before execution, keyed by `(run_id, tool_call_id)`. Managed file operations can be atomic and replay-safe. Arbitrary shell commands cannot always guarantee exactly-once execution after a worker crash, so reconcile sandbox state or fail closed when the outcome is ambiguous.

### 3.5 Resumable Event Stream

SSE is sufficient for server-to-client progress and simpler to resume than a custom WebSocket protocol.

```http
GET /v1/agent-runs/run_456/events
Accept: text/event-stream
Last-Event-ID: 41
```

```text
id: 42
event: tool.started
data: {"run_id":"run_456","tool_call_id":"call_17","tool":"read_file"}
```

- Persist events before publishing; delivery is ordered and at-least-once per run.
- Clients deduplicate by event ID. Reconnection resumes strictly after `Last-Event-ID`.
- Emit `assistant.progress`, `tool.started`, `tool.completed`, `approval.required`, `run.waiting`, and one terminal event.
- Send heartbeats through idle proxies. Redact secrets and store oversized payloads as access-controlled artifacts.
- If the cursor is older than retention, return `410 Gone`; the client fetches `GET /v1/agent-runs/{id}` and reconnects from the returned current cursor.

---



## 4. Key Design Decisions (What Interviewers Probe)



### 4.1 Context Window Management

The agent loop accumulates history fast. After 20 steps with tool outputs, you can easily exceed context limits.

**Strategy: Sliding window + summarization**

```python
if token_count(history) > 0.7 * context_limit:
    old_turns = history[:-recent_k]
    summary = LLM("Summarize what was done and current state: " + old_turns)
    context_projection = [summary_message] + history[-recent_k:]
```

The immutable transcript/event log remains canonical. Summaries are disposable context projections and can be regenerated; never overwrite the audit history with an LLM-generated summary.

**Working memory** (separate from conversation history):

```python
working_memory = {
    "plan": ["1. Read CSV ✅", "2. Compute VWAP ✅", "3. Save output ⬜"],
    "files_modified": ["vwap.py"],
    "files_read": ["trades.csv (cols: timestamp, symbol, price, qty, side)"],
    "errors_encountered": ["KeyError: 'volume' — column is actually 'qty'"],
    "current_step": 3
}
```

Working memory should carry provenance such as workspace revisions and file hashes. Inject it as clearly marked context, not as trusted system instructions; model-proposed memory updates must be validated against durable state.

### 4.2 Cost Control (Tower cares deeply about this)

```python
class BudgetGuard:
    def __init__(self, max_steps=50, max_cost_usd=2.0, max_time_s=300):
        self.max_steps = max_steps
        self.max_cost_usd = max_cost_usd
        self.max_time_s = max_time_s
        self.steps = 0
        self.cost = 0.0
        self.start_time = time.monotonic()
    
    def reserve(self, worst_case_step_cost: float) -> None:
        elapsed = time.monotonic() - self.start_time
        if self.steps >= self.max_steps:
            raise BudgetExceeded("Step limit reached")
        if self.cost + worst_case_step_cost > self.max_cost_usd:
            raise BudgetExceeded("Insufficient remaining cost budget")
        if elapsed >= self.max_time_s:
            raise BudgetExceeded("Timeout")

        self.steps += 1
        self.cost += worst_case_step_cost

    def settle(self, reserved: float, actual: float) -> None:
        self.cost += actual - reserved
```

Persist budget counters and reserve atomically before each model/tool dispatch so worker retries cannot reset or overspend the budget. Provider pricing can change, so a dollar cap is strict only to the reservation granularity and pricing snapshot; token, output, process, and wall-time limits remain hard controls.

**Anti-loop detection**: Detect lack of observable progress, repeated error signatures, unchanged workspace state, and oscillating plans. Repeated calls may be legitimate polling, so the runtime should request user input or stop after a bounded no-progress budget rather than relying only on identical-call detection.

### 4.3 Safety / Sandboxing

- Use rootless containers with syscall filtering, or microVM isolation for higher-risk workloads; Docker alone is not a complete security boundary
- Use a copy-on-write workspace, canonicalized paths, symlink checks, scoped write roots, and CPU/memory/disk/process/time limits
- Deny network by default or use domain-scoped egress. Broker short-lived, least-privilege credentials instead of placing secrets in prompts or a general environment
- Command-name filtering is bypassable through scripts and shell composition. Enforce filesystem, process, network, credential, and resource capabilities outside the LLM
- Treat repository files and tool output as untrusted data. Truncation controls context size—not prompt injection—so also redact secrets, validate outputs, and keep policy enforcement deterministic



### 4.4 Session Persistence (History)

```python
# Store session state in DB
class AgentSession:
    session_id: str
    user_id: str
    workspace_id: str
    current_workspace_revision: str
    history: list[Message]          # durable public transcript
    created_at: datetime
    updated_at: datetime

class AgentRun:
    run_id: str
    session_id: str
    status: str                     # queued | running | waiting_for_approval |
                                    # cancelling | cancelled | completed |
                                    # failed | budget_exhausted
    limits: BudgetLimits
    usage: BudgetState
    base_workspace_revision: str
    result_workspace_revision: str | None
    checkpoint_ref: str | None
    last_event_id: str | None
    terminal_reason: str | None

# A follow-up creates another run in the same session:
# POST /v1/agent-sessions/{session_id}/runs
# {"input": "Also add a cumulative-volume column"}
```

---



## 5. Failure Modes to Discuss


| Failure                                       | Detection                                                          | Mitigation                                                         |
| --------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------------------------------ |
| Infinite loop (agent keeps retrying same fix) | No progress, repeated error signatures, or oscillating state       | Bound no-progress budget, request input, then stop gracefully      |
| Context window overflow                       | Token count tracking                                               | Summarize old turns, keep working memory lean                      |
| Hallucinated file paths / APIs                | Tool returns error                                                 | Agent reads the error and self-corrects (this is the loop working) |
| Runaway cost                                  | Budget guard                                                       | Hard limits + per-step cost tracking                               |
| Unsafe code execution                         | Sandbox/policy telemetry                                           | Rootless isolation plus filesystem/network/process capabilities    |
| Agent goes off-task                           | Acceptance checks and plan/workspace progress                      | Reproject the original goal, request input, or stop                |
| Duplicate submission or worker retry          | Reused idempotency/tool-call key                                   | Return committed result; never repeat managed side effects         |
| Disconnect during streaming                   | Client reconnect with last event ID                                | Replay durable events or recover from run snapshot                 |
| Cancellation during an active process         | Run enters `cancelling`                                            | Kill the complete process tree; atomically choose one terminal state |
| Stale or replayed approval                     | Proposal hash/version mismatch                                     | Reject with `409`; require approval for the new proposal           |
| Concurrent workspace edits                    | Base revision no longer current                                    | Isolate revisions, serialize runs, or reject the conflicting commit |
| Prompt injection / secret exfiltration        | Policy violation or suspicious egress                              | Untrusted-data boundaries, scoped credentials, and deterministic controls |


---



## 6. How to Present This in an Interview

**Start with** the user need ("researchers need to automate repetitive coding tasks").

**Draw the loop** (LLM decision → runtime dispatch → tool result → next LLM decision). Make clear that the orchestrator is code, not another LLM.

**Then contracts**: session vs run, idempotent creation, cancellation, approval decisions, structured tool calls, and resumable events.

**Then go deep on 2-3 design decisions**: context management, cost control, safety — pick based on interviewer's interest.

**End with** failure modes and how you'd monitor this in production.

Total: 5-8 minutes for a clean walkthrough.