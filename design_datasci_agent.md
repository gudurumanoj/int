# Design: Autonomous Data Scientist Agent

## 1. Problem Framing

> "Design an agent that takes a dataset and a question (e.g., 'what drives churn?'), autonomously explores the data, runs analyses, builds models, and delivers an insight report."

**Who & why**: A quant researcher at Tower drops a CSV of trading signals and asks "which features are most predictive of next-day returns?" The agent should do what a junior data scientist would — but in minutes, not days.

**Features to think out loud about:**
- **Autonomous EDA**: profile the data, detect types, find nulls, compute distributions, spot outliers
- **Hypothesis generation**: given the question, propose analyses to run
- **Code execution**: write and run pandas/sklearn/statsmodels code in a sandbox
- **Iterative refinement**: if a model underperforms, try feature engineering, different models
- **Report generation**: produce a structured markdown/HTML report with charts and findings
- **Human checkpoints**: at key decision points (model selection, feature removal), optionally pause for approval
- **Session history**: user can ask follow-ups ("now try XGBoost instead" or "remove outliers and rerun")

---

## 2. Architecture — The Planner-Worker Pattern

Unlike the code agent (single ReAct loop), a data science agent benefits from a **two-level architecture**: a Planner that creates a high-level analysis plan, and a Worker that executes each step.

```
┌───────────────────────────────────────────────────────┐
│                      USER                              │
│  "Which features predict next-day returns?"            │
└──────────────────────┬────────────────────────────────┘
                       ▼
┌───────────────────────────────────────────────────────┐
│                   PLANNER LLM                          │
│  Input: user question + data schema + data profile     │
│  Output: ordered list of analysis steps                │
│                                                        │
│  Plan:                                                 │
│    1. Load & profile data                              │
│    2. Clean: handle nulls, fix dtypes, remove outliers │
│    3. Feature correlation analysis                     │
│    4. Train baseline model (linear regression)         │
│    5. Feature importance via permutation importance    │
│    6. Try tree-based model (XGBoost), compare          │
│    7. Generate report with top features + charts       │
└──────────────────────┬────────────────────────────────┘
                       │ plan (structured JSON)
                       ▼
┌───────────────────────────────────────────────────────┐
│                EXECUTOR (Step Runner)                   │
│                                                        │
│  for runnable step in plan:                            │
│      proposal = CODER_LLM(step, committed_artifacts)   │
│      VALIDATOR.check(proposal, policy, budget)         │
│      if policy.requires_approval(proposal):            │
│          approve_exact_proposal_before_execution()     │
│      result = SANDBOX.run_from_checkpoint(proposal)    │
│      commit(step, result, artifacts, events)           │
│                                                        │
│      # Dynamic, versioned replanning                   │
│      if validated_result changes assumptions:          │
│          plan = PLANNER(replan from committed state)   │
└──────────────────────┬────────────────────────────────┘
                       │
          ┌────────────┼────────────┐
          ▼            ▼            ▼
     ┌─────────┐ ┌─────────┐ ┌──────────┐
     │ SANDBOX │ │ CHART   │ │ REPORT   │
     │(python  │ │ GEN     │ │ BUILDER  │
     │ exec)   │ │(matplotlib│ │(markdown)│
     └─────────┘ └─────────┘ └──────────┘
```

**Why two levels?** A single ReAct loop for data science can meander. A versioned plan bounds the analysis, and the worker proposes one step at a time. Planner and coder may be two calls to the same LLM or separate models; the **orchestrator and sandbox are deterministic application code**, not LLMs. Approval always happens before execution, and retries start from the last committed checkpoint.

---

## 3. Contracts

### 3.1 Session and Run Submission

The session stores durable conversation history. A run freezes one question, dataset version, configuration, and execution attempt. Do not use mutable server filesystem paths as dataset identity.

```http
POST /v1/datasci/sessions
Idempotency-Key: create-signal-session

{"title": "Signal predictiveness"}

201 Created
{"session_id": "sess_123"}
```

```http
POST /v1/datasci/sessions/sess_123/runs
Idempotency-Key: analyze-signal-v3

{
  "question": "Which features are most predictive of next-day returns?",
  "datasets": [{"dataset_id": "ds_456", "version": 3, "sha256": "..."}],
  "parent_run_id": null,
  "config": {
    "target_column": "next_day_return",
    "max_steps": 30,
    "max_cost_usd": 3.0,
    "max_wall_time_seconds": 900,
    "approval_policy": "risk_based",
    "output_format": "markdown_report"
  }
}

202 Accepted
{"run_id": "run_789", "status": "queued", "events_url": "/v1/datasci/runs/run_789/events"}
```

Same idempotency key + same canonical payload returns the existing resource; the same key with a different payload returns `409 Conflict`. Follow-up questions create new runs and never reopen a terminal run.

### 3.2 Run Lifecycle, Cancellation, and Events

```text
queued → running ↔ waiting_for_approval
queued | running | waiting_for_approval → cancelling → cancelled
running → succeeded
any non-terminal state → failed | budget_exhausted
```

```http
GET /v1/datasci/runs/run_789

POST /v1/datasci/runs/run_789/cancel
Idempotency-Key: cancel-run-789
{"reason": "User requested cancellation"}

GET /v1/datasci/runs/run_789/events
Accept: text/event-stream
Last-Event-ID: 124
```

Cancellation is cooperative first and forced after a grace period. Stop queued work, kill the sandbox process tree, reject late artifact commits, and mark already committed artifacts as partial. A compare-and-set transition resolves cancel/complete races.

Events are persisted before publication and carry a monotonically increasing sequence per run. SSE delivery is at-least-once; clients deduplicate by event ID and reconnect with `Last-Event-ID`. If retention has expired, return `410 Gone` and let the client recover from the current run snapshot.

### 3.3 Plan Schema (Planner → Orchestrator)

```python
class AnalysisPlan:
    plan_id: str
    version: int
    parent_version: int | None
    based_on_event_seq: int
    steps: list[AnalysisStep]
    
class AnalysisStep:
    id: str                      # "step_01_eda"
    description: str             # "Load data and compute summary statistics"
    depends_on: list[str]        # ["step_00_load"] — DAG ordering
    step_type: str               # "eda" | "cleaning" | "feature_eng" | "modeling" | "evaluation" | "reporting"
    input_artifact_ids: list[str]
    expected_outputs: list[str]
    acceptance_checks: list[str]
    risk_tags: list[str]         # e.g. target_change, row_drop, package_install
```

Approval policy is computed outside the LLM from risk tags, generated code, user/tenant policy, and predicted effects.

### 3.4 Approval Decision

```python
class ApprovalRequest:
    approval_id: str
    run_id: str
    plan_version: int
    step_id: str
    proposal_hash: str
    summary: str
    risk_tags: list[str]
    expires_at: datetime
```

```http
PUT /v1/datasci/approval-requests/apr_123/decision
Idempotency-Key: decide-apr-123

{
  "decision": "approve",
  "expected_proposal_hash": "sha256:...",
  "comment": "Target and split strategy look correct"
}
```

Supported decisions are `approve`, `reject`, and `request_changes`. Approval executes only the exact proposal hash. Requested changes create a new plan/proposal version requiring fresh approval. Repeated identical decisions are safe; conflicting, stale, expired, or already superseded decisions return `409 Conflict`.

### 3.5 Step Execution Contract

```python
class StepAttemptResult:
    run_id: str
    plan_version: int
    step_id: str
    attempt_id: str
    status: str                    # succeeded | failed | cancelled | timed_out
    code_ref: str
    code_sha256: str
    artifact_refs: list[str]
    log_ref: str | None             # bounded logs live outside the row
    metrics: dict[str, float]
    checkpoint_ref: str | None
    error: dict | None
    started_at: datetime
    finished_at: datetime
```

Artifacts live in object storage with media type, size, hash, and lineage. Duplicate queue delivery may rerun an isolated attempt, but only one result can atomically commit for a `(plan_version, step_id)`.

### 3.6 Final Report Contract

```python
class AnalysisReport:
    title: str
    question: str
    executive_summary: str        # 2-3 sentence answer to the question
    sections: list[ReportSection]
    charts: list[Chart]
    methodology: str              # what models/techniques were used
    limitations: str              # caveats
    code_bundle_ref: str
    reproducibility_manifest_ref: str
    session_id: str
    run_id: str

class ReportSection:
    heading: str
    content: str                  # markdown
    chart_refs: list[str]         # chart IDs embedded in this section
```

The reproducibility manifest records dataset IDs/hashes, image digest, dependency lock, plan/prompt/model versions, random seeds, split hashes, and code/artifact hashes. The result is traceable and rerunnable, though external LLM calls and some numerical kernels may not be bit-for-bit deterministic.

---

## 4. Key Design Decisions

### 4.1 Data Profiling as the First Step (Always)

Before the LLM ever plans an analysis, run a **deterministic profiler**:

```python
def profile_dataset(df):
    return {
        "shape": df.shape,
        "columns": {col: {
            "dtype": str(df[col].dtype),
            "nulls": int(df[col].isnull().sum()),
            "null_pct": round(df[col].isnull().mean() * 100, 1),
            "nunique": int(df[col].nunique()),
            "sample_values": redact_samples(df[col], limit=3),
            "mean": round(df[col].mean(), 4) if is_numeric_dtype(df[col]) else None,
            "std": round(df[col].std(), 4) if is_numeric_dtype(df[col]) else None,
        } for col in df.columns},
        "row_count": len(df),
    }
```

This is **cheap** (no LLM call) and deterministic. It supplies sanitized metadata for an initial plan; execution still receives scoped access to the immutable dataset snapshot. For large data, use bounded scans or seeded samples and record the sampling policy. Detect PII/secrets and do not send raw sample values to the LLM unless policy permits. Record the profiler version, dataset hash, and profile artifact hash.

### 4.2 Dynamic Replanning

The initial plan is a hypothesis. During execution, the agent might discover:
- The target column has 90% nulls → need to redefine the target
- A feature is perfectly correlated with the target → data leakage
- The baseline model already gets R²=0.95 → maybe the task is trivial

```python
class Replanner:
    def should_replan(self, step_result, current_plan):
        triggers = [
            step_result.schema_mismatch,
            step_result.target_invalid,
            step_result.reveals_data_leakage,
            step_result.acceptance_check_failed,
            step_result.retry_budget_exhausted,
        ]
        return any(triggers)
    
    def replan(self, original_question, completed_steps, discovery):
        prompt = f"""
        Original question: {original_question}
        Completed so far: {completed_steps}
        New discovery: {discovery}
        
        Revise the remaining analysis plan.
        """
        return planner_llm(prompt)  # creates immutable plan version N+1
```

Each replan preserves completed immutable outputs, uses compare-and-set against the current plan version, and consumes a bounded replan budget. A high score such as R² = 0.95 is an alert to inspect—not proof of leakage by itself.

### 4.3 Checkpointed Sandbox (Code Continuity)

A durable session is storage, not a durable Python process. Each run gets an ephemeral sandbox. Successful steps commit immutable artifacts and a checkpoint manifest; retries and crash recovery resume from the last committed checkpoint.

```python
class RunSandbox:
    """Optional warm kernel within one run; never the source of truth."""
    
    def __init__(self):
        self.kernel = start_ipython_kernel()
    
    def execute(self, code: str) -> StepAttemptResult:
        result = self.kernel.execute(code)
        artifact_refs = export_allowlisted_outputs(result)
        return StepAttemptResult(
            log_ref=store_bounded_log(result.stdout, result.stderr),
            artifact_refs=artifact_refs,
            checkpoint_ref=commit_checkpoint_manifest(artifact_refs),
            ...
        )
```

Do not serialize arbitrary namespaces or untrusted pickle files. Store allowlisted formats (for example Parquet, JSON, ONNX where appropriate) or rerun deterministic code. A warm kernel is only a performance optimization inside one run.

### 4.4 Chart Generation Strategy

LLM-generated matplotlib code is fragile. Better approach:

```python
# Agent produces structured chart spec, not raw matplotlib code
chart_spec = {
    "type": "bar",
    "title": "Top 10 Features by Importance",
    "x": feature_names[:10],
    "y": importances[:10],
    "xlabel": "Feature",
    "ylabel": "Permutation Importance"
}

# Deterministic renderer (no LLM involved)
def render_chart(spec) -> bytes:
    fig, ax = plt.subplots(figsize=(10, 6))
    if spec["type"] == "bar":
        ax.barh(spec["x"], spec["y"])
    ax.set_title(spec["title"])
    ...
    return fig_to_png(fig)
```

This decouples "what to chart" (LLM decides) from "how to render" (deterministic code). Far more reliable.

### 4.5 Statistical Validity and Leakage

- Classify the question as descriptive, predictive, or causal. “What drives churn?” is causal language; without an identification strategy, report predictive association
- Freeze the target, prediction horizon, decision time, unit of analysis, and primary metric before model search
- Split before imputation, scaling, feature selection, or tuning; fit transforms inside each training fold
- Use grouped splits for repeated entities and chronological splits for temporal data. For next-day returns, use walk-forward or purged cross-validation with an embargo when labels overlap
- Keep a final holdout untouched until model selection finishes; report baselines, uncertainty, and sensitivity across seeds/time windows
- Validate feature availability timestamps, as-of joins, target-derived columns, duplicate entities, survivorship bias, and transaction costs
- Broad feature searches require multiple-hypothesis correction or an explicit false-discovery caveat

Leakage prevention must be enforced by split-aware pipeline APIs, provenance, and runtime checks—not only by prompting the model.

### 4.6 Sandbox and Data Security

- Isolate each run/tenant; deny network and cloud credentials by default
- Mount immutable inputs and a scoped artifact-output directory
- Pin environment image and dependency lock; disallow arbitrary package installation by default
- Enforce CPU, memory, disk, process, output, and wall-time limits
- Validate paths and chart specs, track child processes, and sanitize generated Markdown/HTML
- Cancellation kills the entire process tree and sandbox state expires after the configured TTL

---

## 5. Failure Modes

| Failure | Cause | Mitigation |
|---------|-------|------------|
| Agent trains model on test data | Lack of ML hygiene | Split-aware pipeline API, provenance, and holdout-access policy |
| Agent picks wrong target column | Ambiguous user question | Profiler + confirmation step: "I identified `next_day_return` as the target. Correct?" |
| Generated code imports unavailable library | Sandbox doesn't have it | Pre-install common libs (pandas, sklearn, xgboost, statsmodels); fail gracefully with suggestion |
| Agent spends 20 steps on EDA, never models | Poor planning | Step budget per phase: max 5 steps for EDA, max 10 for modeling |
| Report is incoherent | LLM summarization quality | Structured report template; each section tied to a specific step result |
| Duplicate submission or queue delivery | Client/worker retry | Idempotency key plus atomic step-result commit |
| Worker crashes after artifact upload | Partial commit | Stage artifacts, then atomically commit the manifest |
| Approval is stale or replayed | Proposal changed | Bind decision to proposal hash/version and reject stale decisions |
| Client disconnects during streaming | Network/proxy failure | Durable event sequence and SSE replay |
| Cancellation races with completion | Concurrent state transitions | Compare-and-set one terminal state; reject late commits |
| Statistical false discovery | Broad feature/model search | Correction, untouched holdout, and uncertainty reporting |
| Generated code attempts exfiltration | Malicious data or model output | Default-deny network/credentials and isolated sandbox |

---

## 6. Interview Walkthrough (5-min version)

1. **"The user gives us a dataset and a question."** Show the submission API.
2. **"First, we profile the data deterministically — no LLM needed."** This shows you know not everything needs an LLM.
3. **"The planner creates a structured analysis plan."** Draw the plan as a list.
4. **"The orchestrator validates each proposal and obtains policy-driven approval before execution."**
5. **"Each isolated attempt atomically commits artifacts, a checkpoint, and durable events."**
6. **"Runs are retry-safe, cancellable, and observable through a resumable stream."**
7. **"Unexpected validated findings create a new plan version; temporal/grouped splits and holdout policy protect validity."**
8. **"The report links every claim to evidence and a reproducibility manifest; charts render deterministically from specs."**
