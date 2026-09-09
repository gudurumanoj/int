from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .util import canonical_json, new_id, payload_hash, utc_now


TERMINAL_STATUSES = {"succeeded", "failed", "cancelled", "budget_exhausted"}


class ConflictError(Exception):
    pass


class NotFoundError(Exception):
    pass


class Database:
    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.RLock()

    def connect(self) -> sqlite3.Connection:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        con.execute("PRAGMA foreign_keys = ON")
        con.execute("PRAGMA journal_mode = WAL")
        return con

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS dataset_series (
                    filename TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL UNIQUE,
                    latest_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    path TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (dataset_id, version),
                    UNIQUE (dataset_id, sha256)
                );
                CREATE TABLE IF NOT EXISTS runs (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL REFERENCES sessions(id),
                    question TEXT NOT NULL,
                    datasets_json TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    parent_run_id TEXT,
                    status TEXT NOT NULL,
                    cancellation_requested INTEGER NOT NULL DEFAULT 0,
                    cancellation_reason TEXT,
                    current_plan_version INTEGER,
                    report_artifact_id TEXT,
                    manifest_artifact_id TEXT,
                    error_json TEXT,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS idempotency (
                    scope TEXT NOT NULL,
                    key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    response_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (scope, key)
                );
                CREATE TABLE IF NOT EXISTS events (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, seq)
                );
                CREATE TABLE IF NOT EXISTS plans (
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    version INTEGER NOT NULL,
                    plan_id TEXT NOT NULL,
                    parent_version INTEGER,
                    based_on_event_seq INTEGER NOT NULL,
                    plan_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, version)
                );
                CREATE TABLE IF NOT EXISTS steps (
                    run_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    step_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    proposal_hash TEXT,
                    code_ref TEXT,
                    risk_tags_json TEXT,
                    result_json TEXT,
                    PRIMARY KEY (run_id, plan_version, step_id),
                    FOREIGN KEY (run_id, plan_version)
                        REFERENCES plans(run_id, version)
                );
                CREATE TABLE IF NOT EXISTS approvals (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    plan_version INTEGER NOT NULL,
                    step_id TEXT NOT NULL,
                    proposal_hash TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    risk_tags_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    decision TEXT,
                    comment TEXT,
                    expires_at TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    decided_at TEXT
                );
                CREATE TABLE IF NOT EXISTS artifacts (
                    id TEXT PRIMARY KEY,
                    run_id TEXT NOT NULL REFERENCES runs(id),
                    plan_version INTEGER,
                    step_id TEXT,
                    kind TEXT NOT NULL,
                    path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    partial INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS checkpoints (
                    run_id TEXT NOT NULL,
                    plan_version INTEGER NOT NULL,
                    step_id TEXT NOT NULL,
                    manifest_path TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (run_id, plan_version, step_id)
                );
                """
            )

    def idempotent(
        self,
        scope: str,
        key: str,
        payload: Any,
        operation: Callable[[sqlite3.Connection], tuple[int, dict[str, Any]]],
    ) -> tuple[int, dict[str, Any], bool]:
        digest = payload_hash(payload)
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT * FROM idempotency WHERE scope = ? AND key = ?",
                (scope, key),
            ).fetchone()
            if row:
                if row["request_hash"] != digest:
                    raise ConflictError(
                        "idempotency key was already used with a different payload"
                    )
                return row["status_code"], json.loads(row["response_json"]), True
            status_code, response = operation(con)
            con.execute(
                """
                INSERT INTO idempotency
                    (scope, key, request_hash, status_code, response_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (scope, key, digest, status_code, canonical_json(response), utc_now()),
            )
            return status_code, response, False

    def create_session(
        self, key: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], bool]:
        def operation(con: sqlite3.Connection) -> tuple[int, dict[str, Any]]:
            session_id = new_id("sess")
            con.execute(
                "INSERT INTO sessions (id, title, created_at) VALUES (?, ?, ?)",
                (session_id, payload["title"], utc_now()),
            )
            return 201, {"session_id": session_id}

        return self.idempotent("create_session", key, payload, operation)

    def register_dataset(
        self, filename: str, sha256: str, path: str, size_bytes: int
    ) -> dict[str, Any]:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            series = con.execute(
                "SELECT * FROM dataset_series WHERE filename = ?", (filename,)
            ).fetchone()
            if series:
                existing = con.execute(
                    """
                    SELECT * FROM datasets
                    WHERE dataset_id = ? AND sha256 = ?
                    """,
                    (series["dataset_id"], sha256),
                ).fetchone()
                if existing:
                    return self._dataset_dict(existing)
                dataset_id = series["dataset_id"]
                version = series["latest_version"] + 1
                con.execute(
                    """
                    UPDATE dataset_series SET latest_version = ?
                    WHERE filename = ?
                    """,
                    (version, filename),
                )
            else:
                dataset_id = new_id("ds")
                version = 1
                con.execute(
                    """
                    INSERT INTO dataset_series (filename, dataset_id, latest_version)
                    VALUES (?, ?, ?)
                    """,
                    (filename, dataset_id, version),
                )
            con.execute(
                """
                INSERT INTO datasets
                    (dataset_id, version, filename, sha256, path, size_bytes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (dataset_id, version, filename, sha256, path, size_bytes, utc_now()),
            )
            return {
                "dataset_id": dataset_id,
                "version": version,
                "filename": filename,
                "sha256": sha256,
                "path": path,
                "size_bytes": size_bytes,
            }

    @staticmethod
    def _dataset_dict(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "dataset_id": row["dataset_id"],
            "version": row["version"],
            "filename": row["filename"],
            "sha256": row["sha256"],
            "path": row["path"],
            "size_bytes": row["size_bytes"],
        }

    def get_dataset(
        self, dataset_id: str, version: int, sha256: str | None = None
    ) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM datasets WHERE dataset_id = ? AND version = ?",
                (dataset_id, version),
            ).fetchone()
        if not row:
            raise NotFoundError("dataset version not found")
        result = self._dataset_dict(row)
        normalized = (sha256 or "").removeprefix("sha256:")
        if normalized and normalized != result["sha256"]:
            raise ConflictError("dataset hash does not match immutable version")
        return result

    def create_run(
        self, session_id: str, key: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], bool]:
        def operation(con: sqlite3.Connection) -> tuple[int, dict[str, Any]]:
            if not con.execute(
                "SELECT 1 FROM sessions WHERE id = ?", (session_id,)
            ).fetchone():
                raise NotFoundError("session not found")
            run_id = new_id("run")
            now = utc_now()
            con.execute(
                """
                INSERT INTO runs
                    (id, session_id, question, datasets_json, config_json,
                     parent_run_id, status, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    payload["question"],
                    canonical_json(payload["datasets"]),
                    canonical_json(payload["config"]),
                    payload.get("parent_run_id"),
                    now,
                    now,
                ),
            )
            self._append_event_tx(con, run_id, "run.queued", {"status": "queued"})
            return 202, {
                "run_id": run_id,
                "status": "queued",
                "events_url": f"/v1/datasci/runs/{run_id}/events",
            }

        return self.idempotent(f"create_run:{session_id}", key, payload, operation)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute("SELECT * FROM runs WHERE id = ?", (run_id,)).fetchone()
        if not row:
            raise NotFoundError("run not found")
        result = dict(row)
        result["datasets"] = json.loads(result.pop("datasets_json"))
        result["config"] = json.loads(result.pop("config_json"))
        error_json = result.pop("error_json")
        result["error"] = json.loads(error_json) if error_json else None
        result["cancellation_requested"] = bool(result["cancellation_requested"])
        return result

    def set_running(self, run_id: str) -> bool:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status, cancellation_requested FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if not row:
                raise NotFoundError("run not found")
            if row["cancellation_requested"] or row["status"] in TERMINAL_STATUSES:
                return False
            if row["status"] not in {"queued", "running"}:
                return False
            con.execute(
                "UPDATE runs SET status = 'running', updated_at = ? WHERE id = ?",
                (utc_now(), run_id),
            )
            self._append_event_tx(
                con, run_id, "run.running", {"status": "running"}
            )
            return True

    def request_cancel(
        self, run_id: str, key: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], bool]:
        def operation(con: sqlite3.Connection) -> tuple[int, dict[str, Any]]:
            row = con.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row:
                raise NotFoundError("run not found")
            if row["status"] == "succeeded":
                raise ConflictError("a succeeded run cannot be cancelled")
            if row["status"] not in TERMINAL_STATUSES:
                con.execute(
                    """
                    UPDATE runs
                    SET status = 'cancelling', cancellation_requested = 1,
                        cancellation_reason = ?, updated_at = ?
                    WHERE id = ?
                    """,
                    (payload["reason"], utc_now(), run_id),
                )
                self._append_event_tx(
                    con,
                    run_id,
                    "run.cancelling",
                    {"status": "cancelling", "reason": payload["reason"]},
                )
                con.execute(
                    """
                    UPDATE runs SET status = 'cancelled', updated_at = ?
                    WHERE id = ? AND status = 'cancelling'
                    """,
                    (utc_now(), run_id),
                )
                con.execute(
                    "UPDATE artifacts SET partial = 1 WHERE run_id = ?", (run_id,)
                )
                self._append_event_tx(
                    con, run_id, "run.cancelled", {"status": "cancelled"}
                )
            status = con.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()["status"]
            return 200, {"run_id": run_id, "status": status}

        return self.idempotent(f"cancel:{run_id}", key, payload, operation)

    def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> int:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            return self._append_event_tx(con, run_id, event_type, data)

    @staticmethod
    def _append_event_tx(
        con: sqlite3.Connection,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> int:
        seq = con.execute(
            "SELECT COALESCE(MAX(seq), 0) + 1 FROM events WHERE run_id = ?",
            (run_id,),
        ).fetchone()[0]
        con.execute(
            """
            INSERT INTO events (run_id, seq, event_type, data_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, seq, event_type, canonical_json(data), utc_now()),
        )
        return seq

    def get_events(self, run_id: str, after_seq: int) -> list[dict[str, Any]]:
        self.get_run(run_id)
        with self.connect() as con:
            rows = con.execute(
                """
                SELECT * FROM events WHERE run_id = ? AND seq > ?
                ORDER BY seq
                """,
                (run_id, after_seq),
            ).fetchall()
        return [
            {
                "seq": row["seq"],
                "event_type": row["event_type"],
                "data": json.loads(row["data_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def save_plan(self, run_id: str, plan: dict[str, Any]) -> None:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                """
                INSERT INTO plans
                    (run_id, version, plan_id, parent_version,
                     based_on_event_seq, plan_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    plan["version"],
                    plan["plan_id"],
                    plan.get("parent_version"),
                    plan["based_on_event_seq"],
                    canonical_json(plan),
                    utc_now(),
                ),
            )
            for step in plan["steps"]:
                con.execute(
                    """
                    INSERT INTO steps (run_id, plan_version, step_id, status)
                    VALUES (?, ?, ?, 'pending')
                    """,
                    (run_id, plan["version"], step["id"]),
                )
            con.execute(
                """
                UPDATE runs SET current_plan_version = ?, updated_at = ?
                WHERE id = ?
                """,
                (plan["version"], utc_now(), run_id),
            )
            self._append_event_tx(
                con,
                run_id,
                "plan.created",
                {
                    "plan_id": plan["plan_id"],
                    "version": plan["version"],
                    "step_count": len(plan["steps"]),
                },
            )

    def get_current_plan(self, run_id: str) -> dict[str, Any] | None:
        run = self.get_run(run_id)
        if run["current_plan_version"] is None:
            return None
        with self.connect() as con:
            row = con.execute(
                "SELECT plan_json FROM plans WHERE run_id = ? AND version = ?",
                (run_id, run["current_plan_version"]),
            ).fetchone()
        return json.loads(row["plan_json"]) if row else None

    def get_step(self, run_id: str, version: int, step_id: str) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT * FROM steps
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                """,
                (run_id, version, step_id),
            ).fetchone()
        if not row:
            raise NotFoundError("step not found")
        result = dict(row)
        result["risk_tags"] = (
            json.loads(result.pop("risk_tags_json"))
            if result["risk_tags_json"]
            else []
        )
        result["result"] = (
            json.loads(result.pop("result_json")) if result["result_json"] else None
        )
        return result

    def save_proposal(
        self,
        run_id: str,
        version: int,
        step_id: str,
        proposal_hash: str,
        code_ref: str,
        risk_tags: list[str],
    ) -> None:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                """
                UPDATE steps SET status = 'proposed', proposal_hash = ?,
                    code_ref = ?, risk_tags_json = ?
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                    AND status = 'pending'
                """,
                (
                    proposal_hash,
                    code_ref,
                    canonical_json(risk_tags),
                    run_id,
                    version,
                    step_id,
                ),
            )
            self._append_event_tx(
                con,
                run_id,
                "step.validated",
                {
                    "plan_version": version,
                    "step_id": step_id,
                    "proposal_hash": proposal_hash,
                    "risk_tags": risk_tags,
                },
            )

    def create_approval(
        self,
        run_id: str,
        version: int,
        step_id: str,
        proposal_hash: str,
        summary: str,
        risk_tags: list[str],
        expires_at: str,
    ) -> dict[str, Any]:
        approval_id = new_id("apr")
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            con.execute(
                """
                INSERT INTO approvals
                    (id, run_id, plan_version, step_id, proposal_hash, summary,
                     risk_tags_json, status, expires_at, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, 'pending', ?, ?)
                """,
                (
                    approval_id,
                    run_id,
                    version,
                    step_id,
                    proposal_hash,
                    summary,
                    canonical_json(risk_tags),
                    expires_at,
                    utc_now(),
                ),
            )
            con.execute(
                """
                UPDATE steps SET status = 'waiting_for_approval'
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                """,
                (run_id, version, step_id),
            )
            con.execute(
                """
                UPDATE runs SET status = 'waiting_for_approval', updated_at = ?
                WHERE id = ?
                """,
                (utc_now(), run_id),
            )
            self._append_event_tx(
                con,
                run_id,
                "approval.required",
                {
                    "approval_id": approval_id,
                    "plan_version": version,
                    "step_id": step_id,
                    "proposal_hash": proposal_hash,
                    "risk_tags": risk_tags,
                    "expires_at": expires_at,
                },
            )
        return self.get_approval(approval_id)

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        with self.connect() as con:
            row = con.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
        if not row:
            raise NotFoundError("approval request not found")
        result = dict(row)
        result["risk_tags"] = json.loads(result.pop("risk_tags_json"))
        return result

    def find_step_approval(
        self, run_id: str, version: int, step_id: str
    ) -> dict[str, Any] | None:
        with self.connect() as con:
            row = con.execute(
                """
                SELECT id FROM approvals
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                ORDER BY created_at DESC LIMIT 1
                """,
                (run_id, version, step_id),
            ).fetchone()
        return self.get_approval(row["id"]) if row else None

    def decide_approval(
        self, approval_id: str, key: str, payload: dict[str, Any]
    ) -> tuple[int, dict[str, Any], bool]:
        def operation(con: sqlite3.Connection) -> tuple[int, dict[str, Any]]:
            row = con.execute(
                "SELECT * FROM approvals WHERE id = ?", (approval_id,)
            ).fetchone()
            if not row:
                raise NotFoundError("approval request not found")
            if payload["expected_proposal_hash"] != row["proposal_hash"]:
                raise ConflictError("proposal hash is stale")
            expected_version = payload.get("expected_plan_version")
            if expected_version is not None and expected_version != row["plan_version"]:
                raise ConflictError("plan version is stale")
            if row["status"] != "pending":
                if (
                    row["decision"] == payload["decision"]
                    and row["comment"] == payload.get("comment")
                ):
                    return 200, self._approval_response(dict(row))
                raise ConflictError("approval request was already decided")
            if row["expires_at"] <= utc_now():
                raise ConflictError("approval request has expired")
            current = con.execute(
                """
                SELECT runs.current_plan_version, steps.proposal_hash, steps.status
                FROM runs
                JOIN steps ON steps.run_id = runs.id
                WHERE runs.id = ? AND steps.plan_version = ?
                    AND steps.step_id = ?
                """,
                (row["run_id"], row["plan_version"], row["step_id"]),
            ).fetchone()
            if (
                not current
                or current["current_plan_version"] != row["plan_version"]
                or current["proposal_hash"] != row["proposal_hash"]
                or current["status"] != "waiting_for_approval"
            ):
                raise ConflictError("approval request was superseded")

            decision = payload["decision"]
            now = utc_now()
            con.execute(
                """
                UPDATE approvals SET status = 'decided', decision = ?,
                    comment = ?, decided_at = ? WHERE id = ?
                """,
                (decision, payload.get("comment"), now, approval_id),
            )
            if decision == "approve":
                step_status = "approved"
                run_status = "queued"
            elif decision == "request_changes":
                step_status = "pending"
                run_status = "queued"
            else:
                step_status = "rejected"
                run_status = "failed"
            con.execute(
                """
                UPDATE steps SET status = ?
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                """,
                (step_status, row["run_id"], row["plan_version"], row["step_id"]),
            )
            con.execute(
                """
                UPDATE runs SET status = ?, updated_at = ? WHERE id = ?
                """,
                (run_status, now, row["run_id"]),
            )
            self._append_event_tx(
                con,
                row["run_id"],
                "approval.decided",
                {
                    "approval_id": approval_id,
                    "decision": decision,
                    "plan_version": row["plan_version"],
                    "step_id": row["step_id"],
                    "proposal_hash": row["proposal_hash"],
                },
            )
            updated = dict(row)
            updated.update(
                status="decided",
                decision=decision,
                comment=payload.get("comment"),
                decided_at=now,
            )
            return 200, self._approval_response(updated)

        return self.idempotent(f"approval:{approval_id}", key, payload, operation)

    @staticmethod
    def _approval_response(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "approval_id": row["id"],
            "run_id": row["run_id"],
            "plan_version": row["plan_version"],
            "step_id": row["step_id"],
            "proposal_hash": row["proposal_hash"],
            "status": row["status"],
            "decision": row.get("decision"),
        }

    def commit_step(
        self,
        run_id: str,
        version: int,
        step_id: str,
        result: dict[str, Any],
        artifacts: list[dict[str, Any]],
        checkpoint: dict[str, Any],
    ) -> bool:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            run = con.execute(
                "SELECT status, cancellation_requested FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            step = con.execute(
                """
                SELECT status FROM steps
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                """,
                (run_id, version, step_id),
            ).fetchone()
            if (
                not run
                or not step
                or run["cancellation_requested"]
                or run["status"] in TERMINAL_STATUSES
                or step["status"] == "succeeded"
            ):
                return False
            for artifact in artifacts:
                con.execute(
                    """
                    INSERT INTO artifacts
                        (id, run_id, plan_version, step_id, kind, path, sha256,
                         size_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact["id"],
                        run_id,
                        version,
                        step_id,
                        artifact["kind"],
                        artifact["path"],
                        artifact["sha256"],
                        artifact["size_bytes"],
                        utc_now(),
                    ),
                )
            con.execute(
                """
                INSERT INTO checkpoints
                    (run_id, plan_version, step_id, manifest_path, sha256, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    version,
                    step_id,
                    checkpoint["path"],
                    checkpoint["sha256"],
                    utc_now(),
                ),
            )
            con.execute(
                """
                UPDATE steps SET status = 'succeeded', result_json = ?
                WHERE run_id = ? AND plan_version = ? AND step_id = ?
                """,
                (canonical_json(result), run_id, version, step_id),
            )
            self._append_event_tx(
                con,
                run_id,
                "step.committed",
                {
                    "plan_version": version,
                    "step_id": step_id,
                    "artifact_ids": [item["id"] for item in artifacts],
                    "checkpoint_sha256": checkpoint["sha256"],
                },
            )
            return True

    def list_artifacts(self, run_id: str) -> list[dict[str, Any]]:
        with self.connect() as con:
            rows = con.execute(
                "SELECT * FROM artifacts WHERE run_id = ? ORDER BY created_at",
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def complete_run(
        self, run_id: str, report: dict[str, Any], manifest: dict[str, Any]
    ) -> bool:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status, cancellation_requested FROM runs WHERE id = ?",
                (run_id,),
            ).fetchone()
            if (
                not row
                or row["cancellation_requested"]
                or row["status"] in TERMINAL_STATUSES
            ):
                return False
            for artifact in (report, manifest):
                con.execute(
                    """
                    INSERT INTO artifacts
                        (id, run_id, kind, path, sha256, size_bytes, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        artifact["id"],
                        run_id,
                        artifact["kind"],
                        artifact["path"],
                        artifact["sha256"],
                        artifact["size_bytes"],
                        utc_now(),
                    ),
                )
            con.execute(
                """
                UPDATE runs SET status = 'succeeded', report_artifact_id = ?,
                    manifest_artifact_id = ?, updated_at = ?
                WHERE id = ?
                """,
                (report["id"], manifest["id"], utc_now(), run_id),
            )
            self._append_event_tx(
                con,
                run_id,
                "run.succeeded",
                {
                    "status": "succeeded",
                    "report_artifact_id": report["id"],
                    "manifest_artifact_id": manifest["id"],
                },
            )
            return True

    def fail_run(self, run_id: str, error: dict[str, Any]) -> None:
        with self._lock, self.connect() as con:
            con.execute("BEGIN IMMEDIATE")
            row = con.execute(
                "SELECT status FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if not row or row["status"] in TERMINAL_STATUSES:
                return
            con.execute(
                """
                UPDATE runs SET status = 'failed', error_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (canonical_json(error), utc_now(), run_id),
            )
            self._append_event_tx(
                con, run_id, "run.failed", {"status": "failed", "error": error}
            )
