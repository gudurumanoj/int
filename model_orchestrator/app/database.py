from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from app.domain import ExecutionPlan, ResponseSnapshot, Usage


class IdempotencyConflict(Exception):
    pass


class EventHistoryGone(Exception):
    pass


class Database:
    def __init__(self, path: Path, idempotency_ttl: int, event_retention: int):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, check_same_thread=False)
        self.connection.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.idempotency_ttl = idempotency_ttl
        self.event_retention = event_retention
        self._initialize()

    def _initialize(self) -> None:
        with self.lock, self.connection:
            self.connection.executescript(
                """
                PRAGMA journal_mode = WAL;
                PRAGMA foreign_keys = ON;

                CREATE TABLE IF NOT EXISTS responses (
                    id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    content TEXT,
                    model_id TEXT,
                    routing_metadata_json TEXT NOT NULL,
                    usage_json TEXT NOT NULL,
                    error TEXT,
                    event_cursor INTEGER NOT NULL DEFAULT 0,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );

                CREATE TABLE IF NOT EXISTS idempotency_records (
                    tenant_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    idempotency_key TEXT NOT NULL,
                    payload_hash TEXT NOT NULL,
                    response_id TEXT NOT NULL,
                    expires_at REAL NOT NULL,
                    PRIMARY KEY (tenant_id, endpoint, idempotency_key),
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                );

                CREATE TABLE IF NOT EXISTS events (
                    response_id TEXT NOT NULL,
                    seq INTEGER NOT NULL,
                    event_type TEXT NOT NULL,
                    data_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    PRIMARY KEY (response_id, seq),
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                );

                CREATE TABLE IF NOT EXISTS attempts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    response_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    attempt_number INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    latency_ms REAL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cost_usd REAL NOT NULL DEFAULT 0,
                    error TEXT,
                    created_at REAL NOT NULL,
                    finished_at REAL,
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                );

                CREATE TABLE IF NOT EXISTS cost_reservations (
                    response_id TEXT PRIMARY KEY,
                    reserved_usd REAL NOT NULL,
                    actual_usd REAL NOT NULL DEFAULT 0,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    settled_at REAL,
                    FOREIGN KEY (response_id) REFERENCES responses(id)
                );
                """
            )

    def create_or_replay(
        self,
        tenant_id: str,
        idempotency_key: str,
        payload_hash: str,
        request_json: str,
        plan: ExecutionPlan,
    ) -> tuple[ResponseSnapshot, bool]:
        now = time.time()
        endpoint = "/v1/responses"
        with self.lock, self.connection:
            self.connection.execute(
                "DELETE FROM idempotency_records WHERE expires_at <= ?",
                (now,),
            )
            existing = self.connection.execute(
                """
                SELECT payload_hash, response_id
                FROM idempotency_records
                WHERE tenant_id = ? AND endpoint = ? AND idempotency_key = ?
                """,
                (tenant_id, endpoint, idempotency_key),
            ).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash:
                    raise IdempotencyConflict
                return self._snapshot_locked(existing["response_id"]), True

            response_id = f"resp_{uuid.uuid4().hex}"
            metadata = {
                "schema_version": "1",
                "strategy": plan.strategy,
                "registry_version": plan.registry_version,
                "models_planned": plan.models,
                "quality_gate": plan.quality_gate_id,
                "estimated_max_cost_usd": plan.estimated_max_cost_usd,
                "estimated_p99_latency_ms": plan.estimated_p99_latency_ms,
                "constraint_status": {
                    "cost": "admitted",
                    "deadline": "admitted",
                    "quality": "routing_target_only"
                    if plan.quality_gate_id is None
                    else "gate_available",
                },
            }
            self.connection.execute(
                """
                INSERT INTO responses (
                    id, tenant_id, status, request_json, routing_metadata_json,
                    usage_json, created_at, updated_at
                ) VALUES (?, ?, 'in_progress', ?, ?, ?, ?, ?)
                """,
                (
                    response_id,
                    tenant_id,
                    request_json,
                    json.dumps(metadata, sort_keys=True),
                    Usage().model_dump_json(),
                    now,
                    now,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO idempotency_records (
                    tenant_id, endpoint, idempotency_key, payload_hash,
                    response_id, expires_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    endpoint,
                    idempotency_key,
                    payload_hash,
                    response_id,
                    now + self.idempotency_ttl,
                ),
            )
            self.connection.execute(
                """
                INSERT INTO cost_reservations (
                    response_id, reserved_usd, status, created_at
                ) VALUES (?, ?, 'reserved', ?)
                """,
                (response_id, plan.estimated_max_cost_usd, now),
            )
            self._append_event_locked(
                response_id,
                "response.created",
                {"id": response_id, "status": "in_progress"},
            )
            return self._snapshot_locked(response_id), False

    def get(self, response_id: str) -> ResponseSnapshot | None:
        with self.lock:
            row = self.connection.execute(
                "SELECT id FROM responses WHERE id = ?",
                (response_id,),
            ).fetchone()
            return self._snapshot_locked(response_id) if row else None

    def request_json(self, response_id: str) -> str:
        with self.lock:
            row = self.connection.execute(
                "SELECT request_json FROM responses WHERE id = ?",
                (response_id,),
            ).fetchone()
            if not row:
                raise KeyError(response_id)
            return str(row["request_json"])

    def start_attempt(self, response_id: str, model_id: str, number: int) -> int:
        with self.lock, self.connection:
            cursor = self.connection.execute(
                """
                INSERT INTO attempts (
                    response_id, model_id, attempt_number, status, created_at
                ) VALUES (?, ?, ?, 'running', ?)
                """,
                (response_id, model_id, number, time.time()),
            )
            return int(cursor.lastrowid)

    def finish_attempt(
        self,
        attempt_id: int,
        status: str,
        latency_ms: float,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cost_usd: float = 0,
        error: str | None = None,
    ) -> None:
        with self.lock, self.connection:
            self.connection.execute(
                """
                UPDATE attempts
                SET status = ?, latency_ms = ?, input_tokens = ?,
                    output_tokens = ?, cost_usd = ?, error = ?, finished_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    latency_ms,
                    input_tokens,
                    output_tokens,
                    cost_usd,
                    error,
                    time.time(),
                    attempt_id,
                ),
            )

    def complete(self, response_id: str, content: str, model_id: str) -> None:
        now = time.time()
        with self.lock, self.connection:
            if not self._is_in_progress_locked(response_id):
                return
            usage = self._usage_locked(response_id)
            self.connection.execute(
                """
                UPDATE responses
                SET status = 'completed', content = ?, model_id = ?,
                    usage_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (content, model_id, usage.model_dump_json(), now, response_id),
            )
            self._append_event_locked(
                response_id,
                "response.output_delta",
                {"delta": content, "model": model_id},
            )
            self._append_event_locked(
                response_id,
                "response.completed",
                {"id": response_id, "status": "completed"},
            )
            self._settle_locked(response_id, usage.cost_usd, now)

    def fail(self, response_id: str, error: str) -> None:
        self._finish_terminal(response_id, "failed", "response.failed", error)

    def cancel(self, response_id: str) -> None:
        self._finish_terminal(
            response_id,
            "cancelled",
            "response.cancelled",
            None,
        )

    def _finish_terminal(
        self,
        response_id: str,
        status: str,
        event_type: str,
        error: str | None,
    ) -> None:
        now = time.time()
        with self.lock, self.connection:
            if not self._is_in_progress_locked(response_id):
                return
            usage = self._usage_locked(response_id)
            self.connection.execute(
                """
                UPDATE responses
                SET status = ?, error = ?, usage_json = ?, updated_at = ?
                WHERE id = ?
                """,
                (status, error, usage.model_dump_json(), now, response_id),
            )
            data: dict[str, Any] = {"id": response_id, "status": status}
            if error:
                data["error"] = error
            self._append_event_locked(response_id, event_type, data)
            self._settle_locked(response_id, usage.cost_usd, now)

    def events_after(self, response_id: str, last_id: int) -> list[dict[str, Any]]:
        cutoff = time.time() - self.event_retention
        with self.lock, self.connection:
            response = self.connection.execute(
                "SELECT event_cursor FROM responses WHERE id = ?",
                (response_id,),
            ).fetchone()
            if not response:
                raise KeyError(response_id)
            self.connection.execute(
                "DELETE FROM events WHERE response_id = ? AND created_at < ?",
                (response_id, cutoff),
            )
            bounds = self.connection.execute(
                "SELECT MIN(seq) AS minimum FROM events WHERE response_id = ?",
                (response_id,),
            ).fetchone()
            minimum = bounds["minimum"]
            cursor = int(response["event_cursor"])
            if (minimum is not None and last_id < int(minimum) - 1) or (
                minimum is None and last_id < cursor
            ):
                raise EventHistoryGone
            rows = self.connection.execute(
                """
                SELECT seq, event_type, data_json
                FROM events
                WHERE response_id = ? AND seq > ?
                ORDER BY seq
                """,
                (response_id, last_id),
            ).fetchall()
            return [
                {
                    "id": int(row["seq"]),
                    "event": row["event_type"],
                    "data": json.loads(row["data_json"]),
                }
                for row in rows
            ]

    def _snapshot_locked(self, response_id: str) -> ResponseSnapshot:
        row = self.connection.execute(
            "SELECT * FROM responses WHERE id = ?",
            (response_id,),
        ).fetchone()
        if not row:
            raise KeyError(response_id)
        return ResponseSnapshot(
            id=row["id"],
            status=row["status"],
            content=row["content"],
            model=row["model_id"],
            routing_metadata=json.loads(row["routing_metadata_json"]),
            usage=Usage.model_validate_json(row["usage_json"]),
            error=row["error"],
            event_cursor=int(row["event_cursor"]),
        )

    def _append_event_locked(
        self,
        response_id: str,
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        row = self.connection.execute(
            "SELECT event_cursor FROM responses WHERE id = ?",
            (response_id,),
        ).fetchone()
        sequence = int(row["event_cursor"]) + 1
        now = time.time()
        self.connection.execute(
            "UPDATE responses SET event_cursor = ?, updated_at = ? WHERE id = ?",
            (sequence, now, response_id),
        )
        self.connection.execute(
            """
            INSERT INTO events (response_id, seq, event_type, data_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (response_id, sequence, event_type, json.dumps(data), now),
        )

    def _usage_locked(self, response_id: str) -> Usage:
        row = self.connection.execute(
            """
            SELECT
                COALESCE(SUM(input_tokens), 0) AS input_tokens,
                COALESCE(SUM(output_tokens), 0) AS output_tokens,
                COALESCE(SUM(cost_usd), 0) AS cost_usd
            FROM attempts
            WHERE response_id = ?
            """,
            (response_id,),
        ).fetchone()
        input_tokens = int(row["input_tokens"])
        output_tokens = int(row["output_tokens"])
        return Usage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=round(float(row["cost_usd"]), 9),
        )

    def _is_in_progress_locked(self, response_id: str) -> bool:
        row = self.connection.execute(
            "SELECT status FROM responses WHERE id = ?",
            (response_id,),
        ).fetchone()
        return bool(row and row["status"] == "in_progress")

    def _settle_locked(self, response_id: str, actual: float, now: float) -> None:
        self.connection.execute(
            """
            UPDATE cost_reservations
            SET actual_usd = ?, status = 'settled', settled_at = ?
            WHERE response_id = ?
            """,
            (actual, now, response_id),
        )

    def close(self) -> None:
        with self.lock:
            self.connection.close()
