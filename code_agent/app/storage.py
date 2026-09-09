from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterator


TERMINAL_STATUSES = {
    "cancelled",
    "completed",
    "failed",
    "budget_exhausted",
}


class StorageError(Exception):
    pass


class NotFound(StorageError):
    pass


class Conflict(StorageError):
    pass


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def value_hash(value: Any) -> str:
    return "sha256:" + hashlib.sha256(
        canonical_json(value).encode("utf-8")
    ).hexdigest()


def _identifier(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class Storage:
    """Small synchronous SQLite store.

    Operations are short and guarded by one process-local lock. This keeps the
    transactional rules visible for the demo while SQLite handles durability.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._connection: sqlite3.Connection | None = None
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(
            self.path,
            check_same_thread=False,
            isolation_level=None,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA journal_mode = WAL")
        connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                workspace_id TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS runs (
                run_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL REFERENCES sessions(session_id),
                input TEXT NOT NULL,
                base_workspace_revision TEXT NOT NULL,
                config_json TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                terminal_reason TEXT,
                cancel_reason TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS idempotency_records (
                scope TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                payload_hash TEXT NOT NULL,
                response_json TEXT NOT NULL,
                status_code INTEGER NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (scope, idempotency_key)
            );

            CREATE TABLE IF NOT EXISTS events (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                sequence INTEGER NOT NULL,
                event_type TEXT NOT NULL,
                data_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                PRIMARY KEY (run_id, sequence)
            );

            CREATE TABLE IF NOT EXISTS approvals (
                approval_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                tool_call_id TEXT NOT NULL,
                arguments_hash TEXT NOT NULL,
                proposal_json TEXT NOT NULL,
                risk_summary TEXT NOT NULL,
                status TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                decision_key TEXT,
                decision_payload_hash TEXT,
                comment TEXT,
                decided_at TEXT,
                UNIQUE (run_id, tool_call_id, arguments_hash)
            );

            CREATE TABLE IF NOT EXISTS tool_calls (
                run_id TEXT NOT NULL REFERENCES runs(run_id),
                tool_call_id TEXT NOT NULL,
                name TEXT NOT NULL,
                arguments_json TEXT NOT NULL,
                arguments_hash TEXT NOT NULL,
                status TEXT NOT NULL,
                result_json TEXT,
                started_at TEXT,
                finished_at TEXT,
                PRIMARY KEY (run_id, tool_call_id)
            );
            """
        )
        self._connection = connection

    def close(self) -> None:
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Storage has not been initialized")
        return self._connection

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            connection = self.connection
            connection.execute("BEGIN IMMEDIATE")
            try:
                yield connection
            except BaseException:
                connection.rollback()
                raise
            else:
                connection.commit()

    @staticmethod
    def _checked_idempotency(
        connection: sqlite3.Connection,
        scope: str,
        key: str,
        payload: Any,
    ) -> tuple[dict[str, Any], int] | None:
        digest = value_hash(payload)
        row = connection.execute(
            """
            SELECT payload_hash, response_json, status_code
            FROM idempotency_records
            WHERE scope = ? AND idempotency_key = ?
            """,
            (scope, key),
        ).fetchone()
        if row is None:
            return None
        if row["payload_hash"] != digest:
            raise Conflict("Idempotency key was already used with another payload")
        return json.loads(row["response_json"]), int(row["status_code"])

    @staticmethod
    def _save_idempotency(
        connection: sqlite3.Connection,
        scope: str,
        key: str,
        payload: Any,
        response: dict[str, Any],
        status_code: int,
    ) -> None:
        connection.execute(
            """
            INSERT INTO idempotency_records (
                scope, idempotency_key, payload_hash, response_json,
                status_code, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                scope,
                key,
                value_hash(payload),
                canonical_json(response),
                status_code,
                utc_now(),
            ),
        )

    def lookup_idempotency(
        self, scope: str, key: str, payload: Any
    ) -> tuple[dict[str, Any], int] | None:
        with self._lock:
            return self._checked_idempotency(
                self.connection, scope, key, payload
            )

    def save_idempotency(
        self,
        scope: str,
        key: str,
        payload: Any,
        response: dict[str, Any],
        status_code: int,
    ) -> None:
        with self._transaction() as connection:
            existing = self._checked_idempotency(
                connection, scope, key, payload
            )
            if existing is None:
                self._save_idempotency(
                    connection,
                    scope,
                    key,
                    payload,
                    response,
                    status_code,
                )

    def create_session(
        self, key: str, payload: dict[str, Any]
    ) -> tuple[dict[str, Any], int, bool]:
        scope = "POST:/v1/agent-sessions"
        with self._transaction() as connection:
            existing = self._checked_idempotency(
                connection, scope, key, payload
            )
            if existing is not None:
                return existing[0], existing[1], True
            now = utc_now()
            session_id = _identifier("sess")
            connection.execute(
                """
                INSERT INTO sessions (
                    session_id, workspace_id, status, created_at, updated_at
                ) VALUES (?, ?, 'idle', ?, ?)
                """,
                (session_id, payload["workspace_id"], now, now),
            )
            response = {"session_id": session_id, "status": "idle"}
            self._save_idempotency(
                connection, scope, key, payload, response, 201
            )
            return response, 201, False

    def create_run(
        self,
        session_id: str,
        key: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], int, bool]:
        scope = f"POST:/v1/agent-sessions/{session_id}/runs"
        with self._transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM sessions WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                is None
            ):
                raise NotFound("Session not found")
            existing = self._checked_idempotency(
                connection, scope, key, payload
            )
            if existing is not None:
                return existing[0], existing[1], True
            now = utc_now()
            run_id = _identifier("run")
            connection.execute(
                """
                INSERT INTO runs (
                    run_id, session_id, input, base_workspace_revision,
                    config_json, status, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    run_id,
                    session_id,
                    payload["input"],
                    payload["base_workspace_revision"],
                    canonical_json(payload["config"]),
                    now,
                    now,
                ),
            )
            self._append_event_tx(
                connection,
                run_id,
                "run.queued",
                {"run_id": run_id, "status": "queued"},
            )
            response = {
                "run_id": run_id,
                "status": "queued",
                "status_url": f"/v1/agent-runs/{run_id}",
                "events_url": f"/v1/agent-runs/{run_id}/events",
            }
            self._save_idempotency(
                connection, scope, key, payload, response, 202
            )
            return response, 202, False

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        if row is None:
            raise NotFound("Run not found")
        result = dict(row)
        result["config"] = json.loads(result.pop("config_json"))
        raw_result = result.pop("result_json")
        result["result"] = json.loads(raw_result) if raw_result else None
        return result

    def transition_to_running(self, run_id: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status = 'running', updated_at = ?
                WHERE run_id = ? AND status = 'queued'
                """,
                (utc_now(), run_id),
            )
            if cursor.rowcount:
                self._append_event_tx(
                    connection,
                    run_id,
                    "run.started",
                    {"run_id": run_id, "status": "running"},
                )
            return bool(cursor.rowcount)

    def complete_run(self, run_id: str, answer: str) -> bool:
        result = {"answer": answer}
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'completed', result_json = ?, updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (canonical_json(result), utc_now(), run_id),
            )
            if cursor.rowcount:
                self._append_event_tx(
                    connection,
                    run_id,
                    "run.completed",
                    {"run_id": run_id, "status": "completed", **result},
                )
            return bool(cursor.rowcount)

    def fail_run(self, run_id: str, reason: str) -> bool:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs
                SET status = 'failed', terminal_reason = ?, updated_at = ?
                WHERE run_id = ? AND status NOT IN (
                    'completed', 'cancelled', 'failed', 'budget_exhausted'
                )
                """,
                (reason, utc_now(), run_id),
            )
            if cursor.rowcount:
                self._append_event_tx(
                    connection,
                    run_id,
                    "run.failed",
                    {"run_id": run_id, "status": "failed", "reason": reason},
                )
            return bool(cursor.rowcount)

    def request_cancel(self, run_id: str, reason: str) -> dict[str, Any]:
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise NotFound("Run not found")
            status = str(row["status"])
            if status in TERMINAL_STATUSES or status == "cancelling":
                return {"run_id": run_id, "status": status}
            connection.execute(
                """
                UPDATE runs SET status = 'cancelling', cancel_reason = ?,
                    updated_at = ? WHERE run_id = ?
                """,
                (reason, utc_now(), run_id),
            )
            connection.execute(
                """
                UPDATE approvals SET status = 'revoked'
                WHERE run_id = ? AND status = 'pending'
                """,
                (run_id,),
            )
            self._append_event_tx(
                connection,
                run_id,
                "run.cancelling",
                {"run_id": run_id, "status": "cancelling", "reason": reason},
            )
            return {"run_id": run_id, "status": "cancelling"}

    def mark_cancelled(self, run_id: str) -> dict[str, Any]:
        with self._transaction() as connection:
            cursor = connection.execute(
                """
                UPDATE runs SET status = 'cancelled', updated_at = ?
                WHERE run_id = ? AND status = 'cancelling'
                """,
                (utc_now(), run_id),
            )
            if cursor.rowcount:
                self._append_event_tx(
                    connection,
                    run_id,
                    "run.cancelled",
                    {"run_id": run_id, "status": "cancelled"},
                )
            row = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise NotFound("Run not found")
            return {"run_id": run_id, "status": str(row["status"])}

    def _append_event_tx(
        self,
        connection: sqlite3.Connection,
        run_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> int:
        row = connection.execute(
            """
            SELECT COALESCE(MAX(sequence), 0) + 1 AS next_sequence
            FROM events WHERE run_id = ?
            """,
            (run_id,),
        ).fetchone()
        sequence = int(row["next_sequence"])
        connection.execute(
            """
            INSERT INTO events (
                run_id, sequence, event_type, data_json, created_at
            ) VALUES (?, ?, ?, ?, ?)
            """,
            (run_id, sequence, event_type, canonical_json(data), utc_now()),
        )
        return sequence

    def append_event(
        self, run_id: str, event_type: str, data: dict[str, Any]
    ) -> int:
        with self._transaction() as connection:
            if (
                connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is None
            ):
                raise NotFound("Run not found")
            return self._append_event_tx(
                connection, run_id, event_type, data
            )

    def list_events_after(
        self, run_id: str, after_sequence: int
    ) -> list[dict[str, Any]]:
        with self._lock:
            if (
                self.connection.execute(
                    "SELECT 1 FROM runs WHERE run_id = ?", (run_id,)
                ).fetchone()
                is None
            ):
                raise NotFound("Run not found")
            rows = self.connection.execute(
                """
                SELECT sequence, event_type, data_json, created_at
                FROM events
                WHERE run_id = ? AND sequence > ?
                ORDER BY sequence
                """,
                (run_id, after_sequence),
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "event_type": row["event_type"],
                "data": json.loads(row["data_json"]),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def create_approval(
        self,
        run_id: str,
        tool_call_id: str,
        arguments: dict[str, Any],
        risk_summary: str,
        expires_at: str,
    ) -> dict[str, Any]:
        digest = value_hash(arguments)
        with self._transaction() as connection:
            existing = connection.execute(
                """
                SELECT * FROM approvals
                WHERE run_id = ? AND tool_call_id = ? AND arguments_hash = ?
                """,
                (run_id, tool_call_id, digest),
            ).fetchone()
            if existing is not None:
                return dict(existing)
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
            if run is None:
                raise NotFound("Run not found")
            if run["status"] != "running":
                raise Conflict("Run is not active")
            approval_id = _identifier("apr")
            connection.execute(
                """
                INSERT INTO approvals (
                    approval_id, run_id, tool_call_id, arguments_hash,
                    proposal_json, risk_summary, status, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)
                """,
                (
                    approval_id,
                    run_id,
                    tool_call_id,
                    digest,
                    canonical_json(arguments),
                    risk_summary,
                    expires_at,
                ),
            )
            connection.execute(
                """
                UPDATE runs SET status = 'waiting_for_approval', updated_at = ?
                WHERE run_id = ? AND status = 'running'
                """,
                (utc_now(), run_id),
            )
            self._append_event_tx(
                connection,
                run_id,
                "approval.required",
                {
                    "run_id": run_id,
                    "approval_id": approval_id,
                    "tool_call_id": tool_call_id,
                    "arguments_hash": digest,
                    "risk_summary": risk_summary,
                    "expires_at": expires_at,
                },
            )
            self._append_event_tx(
                connection,
                run_id,
                "run.waiting",
                {"run_id": run_id, "status": "waiting_for_approval"},
            )
            return {
                "approval_id": approval_id,
                "run_id": run_id,
                "tool_call_id": tool_call_id,
                "arguments_hash": digest,
                "status": "pending",
                "expires_at": expires_at,
            }

    def get_approval(self, approval_id: str) -> dict[str, Any]:
        with self._lock:
            row = self.connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise NotFound("Approval request not found")
        result = dict(row)
        result["proposal"] = json.loads(result.pop("proposal_json"))
        return result

    def decide_approval(
        self,
        approval_id: str,
        idempotency_key: str,
        payload: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        payload_digest = value_hash(payload)
        with self._transaction() as connection:
            row = connection.execute(
                "SELECT * FROM approvals WHERE approval_id = ?",
                (approval_id,),
            ).fetchone()
            if row is None:
                raise NotFound("Approval request not found")
            if row["status"] in {"allow_once", "deny"}:
                if row["decision_payload_hash"] == payload_digest:
                    return {
                        "approval_id": approval_id,
                        "run_id": row["run_id"],
                        "decision": row["status"],
                    }, True
                raise Conflict("Approval already has a different decision")
            if row["status"] != "pending":
                raise Conflict("Approval is stale or revoked")
            if payload["expected_arguments_hash"] != row["arguments_hash"]:
                raise Conflict("Tool arguments changed; a new approval is required")
            if datetime.fromisoformat(row["expires_at"]) <= datetime.now(UTC):
                raise Conflict("Approval request has expired")
            run = connection.execute(
                "SELECT status FROM runs WHERE run_id = ?", (row["run_id"],)
            ).fetchone()
            if run is None or run["status"] != "waiting_for_approval":
                raise Conflict("Approval is stale for the current run state")
            decision = payload["decision"]
            now = utc_now()
            connection.execute(
                """
                UPDATE approvals
                SET status = ?, decision_key = ?, decision_payload_hash = ?,
                    comment = ?, decided_at = ?
                WHERE approval_id = ? AND status = 'pending'
                """,
                (
                    decision,
                    idempotency_key,
                    payload_digest,
                    payload.get("comment"),
                    now,
                    approval_id,
                ),
            )
            connection.execute(
                """
                UPDATE runs SET status = 'running', updated_at = ?
                WHERE run_id = ? AND status = 'waiting_for_approval'
                """,
                (now, row["run_id"]),
            )
            response = {
                "approval_id": approval_id,
                "run_id": row["run_id"],
                "decision": decision,
            }
            self._append_event_tx(
                connection,
                row["run_id"],
                "approval.decided",
                response,
            )
            return response, False

    def begin_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
        name: str,
        arguments: dict[str, Any],
    ) -> tuple[dict[str, Any], bool]:
        digest = value_hash(arguments)
        with self._transaction() as connection:
            row = connection.execute(
                """
                SELECT * FROM tool_calls
                WHERE run_id = ? AND tool_call_id = ?
                """,
                (run_id, tool_call_id),
            ).fetchone()
            if row is not None:
                if (
                    row["name"] != name
                    or row["arguments_hash"] != digest
                ):
                    raise Conflict(
                        "Tool call ID was reused with different arguments"
                    )
                result = (
                    json.loads(row["result_json"])
                    if row["result_json"]
                    else None
                )
                return {"status": row["status"], "result": result}, True
            connection.execute(
                """
                INSERT INTO tool_calls (
                    run_id, tool_call_id, name, arguments_json,
                    arguments_hash, status
                ) VALUES (?, ?, ?, ?, ?, 'pending')
                """,
                (
                    run_id,
                    tool_call_id,
                    name,
                    canonical_json(arguments),
                    digest,
                ),
            )
            return {"status": "pending", "result": None}, False

    def start_tool_call(
        self, run_id: str, tool_call_id: str, name: str
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE tool_calls SET status = 'running', started_at = ?
                WHERE run_id = ? AND tool_call_id = ? AND status = 'pending'
                """,
                (utc_now(), run_id, tool_call_id),
            )
            self._append_event_tx(
                connection,
                run_id,
                "tool.started",
                {
                    "run_id": run_id,
                    "tool_call_id": tool_call_id,
                    "tool": name,
                },
            )

    def finish_tool_call(
        self,
        run_id: str,
        tool_call_id: str,
        result: dict[str, Any],
    ) -> None:
        with self._transaction() as connection:
            connection.execute(
                """
                UPDATE tool_calls
                SET status = ?, result_json = ?, finished_at = ?
                WHERE run_id = ? AND tool_call_id = ?
                """,
                (
                    result["status"],
                    canonical_json(result),
                    utc_now(),
                    run_id,
                    tool_call_id,
                ),
            )
            self._append_event_tx(
                connection,
                run_id,
                "tool.completed",
                {
                    "run_id": run_id,
                    "tool_call_id": tool_call_id,
                    "tool": result["tool_name"],
                    "status": result["status"],
                    "exit_code": result.get("exit_code"),
                    "error_code": result.get("error_code"),
                },
            )
