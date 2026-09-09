from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from .db import Database, TERMINAL_STATUSES
from .llm import PLANNER_PROMPT_VERSION, WORKER_PROMPT_VERSION
from .models import AnalysisPlan, WorkerProposal
from .policy import (
    requires_approval,
    statistical_validity_checklist,
    validate_proposal,
)
from .runner import StepRunner
from .storage import Storage
from .util import new_id, payload_hash, sha256_bytes


class Orchestrator:
    """Deterministic control plane around planner, worker, policy, and runner."""

    def __init__(
        self,
        db: Database,
        storage: Storage,
        llm: Any,
        runner: StepRunner,
    ):
        self.db = db
        self.storage = storage
        self.llm = llm
        self.runner = runner
        self._locks: dict[str, asyncio.Lock] = {}

    async def execute_run(self, run_id: str) -> None:
        lock = self._locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            try:
                await self._execute(run_id)
            except Exception as exc:
                self.db.fail_run(
                    run_id,
                    {"type": type(exc).__name__, "message": str(exc)},
                )

    async def _execute(self, run_id: str) -> None:
        if not self.db.set_running(run_id):
            return
        run = self.db.get_run(run_id)
        datasets = [
            self.db.get_dataset(
                item["dataset_id"], item["version"], item["sha256"]
            )
            for item in run["datasets"]
        ]
        profiles = []
        for dataset in datasets:
            profile = self.storage.profile_csv(dataset["path"])
            profiles.append(
                {
                    "dataset_id": dataset["dataset_id"],
                    "version": dataset["version"],
                    "sha256": dataset["sha256"],
                    "profile": profile,
                    "profile_sha256": payload_hash(profile),
                    "profiler_version": "csv-profiler-v1",
                }
            )

        all_columns = {
            name
            for item in profiles
            for name in item["profile"]["columns"].keys()
        }
        validity = statistical_validity_checklist(run["config"], all_columns)
        profile_seq = self.db.append_event(
            run_id,
            "dataset.profiled",
            {"profiles": profiles, "statistical_validity": validity},
        )

        plan_data = self.db.get_current_plan(run_id)
        if plan_data is None:
            raw_plan = await self.llm.plan(
                run["question"],
                datasets,
                profiles,
                run["config"],
                profile_seq,
            )
            plan = (
                raw_plan
                if isinstance(raw_plan, AnalysisPlan)
                else AnalysisPlan.model_validate(raw_plan)
            )
            if len(plan.steps) > run["config"]["max_steps"]:
                raise ValueError("planner exceeded max_steps")
            self.db.save_plan(run_id, plan.model_dump(mode="json"))
        else:
            plan = AnalysisPlan.model_validate(plan_data)

        for step in plan.steps:
            latest = self.db.get_run(run_id)
            if (
                latest["cancellation_requested"]
                or latest["status"] in TERMINAL_STATUSES
            ):
                return
            step_state = self.db.get_step(run_id, plan.version, step.id)
            if step_state["status"] == "succeeded":
                continue
            if step_state["status"] == "waiting_for_approval":
                return
            if step_state["status"] == "rejected":
                self.db.fail_run(
                    run_id,
                    {"type": "approval_rejected", "step_id": step.id},
                )
                return

            code_ref: dict[str, Any]
            if step_state["status"] == "pending":
                raw_proposal = await self.llm.propose_step(
                    step,
                    datasets[0]["path"],
                    self.db.list_artifacts(run_id),
                    run["config"],
                )
                proposal = (
                    raw_proposal
                    if isinstance(raw_proposal, WorkerProposal)
                    else WorkerProposal.model_validate(raw_proposal)
                )
                validation = validate_proposal(step, proposal, run["config"])
                code_ref = self.storage.save_code(
                    run_id, plan.version, step.id, proposal.code
                )
                proposal_hash = f"sha256:{sha256_bytes(proposal.code.encode('utf-8'))}"
                self.db.save_proposal(
                    run_id,
                    plan.version,
                    step.id,
                    proposal_hash,
                    code_ref["path"],
                    validation.risk_tags,
                )
                self.db.append_event(
                    run_id,
                    "policy.checked",
                    {
                        "plan_version": plan.version,
                        "step_id": step.id,
                        "checks": validation.checks,
                        "warnings": validation.warnings,
                        "requires_approval": requires_approval(
                            validation.risk_tags,
                            run["config"]["approval_policy"],
                        ),
                    },
                )
                if requires_approval(
                    validation.risk_tags, run["config"]["approval_policy"]
                ):
                    expires_at = (
                        datetime.now(UTC) + timedelta(hours=24)
                    ).isoformat()
                    self.db.create_approval(
                        run_id,
                        plan.version,
                        step.id,
                        proposal_hash,
                        step.description,
                        validation.risk_tags,
                        expires_at,
                    )
                    return
                step_state = self.db.get_step(run_id, plan.version, step.id)
            else:
                code_path = Path(step_state["code_ref"])
                code = code_path.read_text(encoding="utf-8")
                code_ref = {
                    "id": new_id("code"),
                    "kind": "code",
                    "path": str(code_path),
                    "sha256": sha256_bytes(code.encode("utf-8")),
                    "size_bytes": len(code.encode("utf-8")),
                }

            code = Path(code_ref["path"]).read_text(encoding="utf-8")
            expected_hash = step_state.get("proposal_hash") or (
                f"sha256:{sha256_bytes(code.encode('utf-8'))}"
            )
            if expected_hash != f"sha256:{sha256_bytes(code.encode('utf-8'))}":
                raise ValueError("stored proposal no longer matches its approval hash")
            validate_proposal(
                step,
                WorkerProposal(
                    code=code, risk_tags=step_state.get("risk_tags", [])
                ),
                run["config"],
            )
            self.db.append_event(
                run_id,
                "step.started",
                {
                    "plan_version": plan.version,
                    "step_id": step.id,
                    "proposal_hash": expected_hash,
                },
            )
            runner_result = await self.runner.run(
                run_id,
                plan.version,
                step.id,
                code,
                datasets[0]["path"],
            )
            if runner_result.status != "succeeded":
                self.db.fail_run(
                    run_id,
                    {
                        "type": "step_execution_failed",
                        "step_id": step.id,
                        "runner_status": runner_result.status,
                        "detail": runner_result.error,
                    },
                )
                return
            artifacts, checkpoint = self.storage.save_step_artifacts(
                run_id,
                plan.version,
                step.id,
                runner_result.artifacts,
                code_ref,
            )
            committed = self.db.commit_step(
                run_id,
                plan.version,
                step.id,
                {
                    "status": runner_result.status,
                    "stdout": runner_result.stdout[-20_000:],
                    "metrics": runner_result.metrics,
                    "proposal_hash": expected_hash,
                    "artifact_refs": [
                        {
                            "id": item["id"],
                            "path": item["path"],
                            "sha256": item["sha256"],
                        }
                        for item in artifacts
                    ],
                },
                artifacts,
                checkpoint,
            )
            if not committed:
                return

        await self._finish_run(run, plan, datasets, validity)

    async def _finish_run(
        self,
        run: dict[str, Any],
        plan: AnalysisPlan,
        datasets: list[dict[str, Any]],
        validity: dict[str, list[str]],
    ) -> None:
        artifacts = self.db.list_artifacts(run["id"])
        checklist_lines = [
            *(f"- [x] {item}" for item in validity["checks"]),
            *(f"- [!] {item}" for item in validity["warnings"]),
        ]
        report = "\n".join(
            [
                f"# Analysis report: {run['question']}",
                "",
                "## Executive summary",
                "The planned analysis steps completed. This educational demo records "
                "outputs and provenance; inspect the linked artifacts before relying "
                "on any substantive conclusion.",
                "",
                "## Statistical-validity checklist",
                *checklist_lines,
                "",
                "## Evidence artifacts",
                *(
                    f"- `{item['kind']}`: `{item['path']}` "
                    f"(sha256 `{item['sha256']}`)"
                    for item in artifacts
                ),
                "",
                "## Methodology",
                f"Executed version {plan.version} of plan `{plan.plan_id}` in "
                "topological order with deterministic policy checks before each step.",
                "",
                "## Limitations",
                "Associations are not causal effects. This demo does not implement "
                "full AutoML, causal identification, or statistical correction.",
                "",
            ]
        )
        pyproject = Path(__file__).parents[1] / "pyproject.toml"
        manifest_data = {
            "manifest_version": 1,
            "run_id": run["id"],
            "session_id": run["session_id"],
            "datasets": [
                {
                    "dataset_id": item["dataset_id"],
                    "version": item["version"],
                    "sha256": item["sha256"],
                }
                for item in datasets
            ],
            "sandbox_image": self.runner.image,
            "dependency_lock_ref": {
                "path": str(pyproject),
                "sha256": sha256_bytes(pyproject.read_bytes()),
            },
            "model": getattr(self.llm, "model", "fake-test-model"),
            "prompt_refs": [PLANNER_PROMPT_VERSION, WORKER_PROMPT_VERSION],
            "plan_ref": {
                "plan_id": plan.plan_id,
                "version": plan.version,
                "sha256": payload_hash(plan.model_dump(mode="json")),
            },
            "random_seed": run["config"]["random_seed"],
            "split_hashes": [],
            "statistical_validity": validity,
            "code_and_artifact_hashes": [
                {
                    "artifact_id": item["id"],
                    "kind": item["kind"],
                    "sha256": item["sha256"],
                }
                for item in artifacts
            ],
        }
        report_ref, manifest_ref = self.storage.save_final(
            run["id"], report, manifest_data
        )
        self.db.complete_run(run["id"], report_ref, manifest_ref)

    async def cancel(self, run_id: str) -> None:
        await self.runner.cancel(run_id)
