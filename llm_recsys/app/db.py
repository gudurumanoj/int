from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Database:
    def __init__(self, path: Path):
        self.path = path

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=5)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.executescript(
                """
                PRAGMA journal_mode = WAL;

                CREATE TABLE IF NOT EXISTS idempotency_records (
                    tenant_id TEXT NOT NULL,
                    endpoint TEXT NOT NULL,
                    idem_key TEXT NOT NULL,
                    request_hash TEXT NOT NULL,
                    response_json TEXT NOT NULL,
                    status_code INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, endpoint, idem_key)
                );

                CREATE TABLE IF NOT EXISTS recommendation_sets (
                    tenant_id TEXT NOT NULL,
                    set_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    expires_at TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    route TEXT NOT NULL,
                    degraded INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, set_id)
                );

                CREATE TABLE IF NOT EXISTS recommendation_items (
                    tenant_id TEXT NOT NULL,
                    set_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    reason TEXT NOT NULL,
                    tracking_token TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, set_id, item_id),
                    UNIQUE (tenant_id, set_id, position),
                    FOREIGN KEY (tenant_id, set_id)
                        REFERENCES recommendation_sets (tenant_id, set_id)
                );

                CREATE TABLE IF NOT EXISTS feedback_events (
                    tenant_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    set_id TEXT NOT NULL,
                    item_id TEXT NOT NULL,
                    position INTEGER NOT NULL,
                    action TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    client_timestamp TEXT NOT NULL,
                    dwell_ms INTEGER,
                    request_hash TEXT NOT NULL,
                    request_json TEXT NOT NULL,
                    received_at TEXT NOT NULL,
                    schema_version INTEGER NOT NULL,
                    PRIMARY KEY (tenant_id, event_id),
                    FOREIGN KEY (tenant_id, set_id)
                        REFERENCES recommendation_sets (tenant_id, set_id)
                );
                """
            )

    def get_idempotency(
        self,
        tenant_id: str,
        endpoint: str,
        idem_key: str,
        now: str,
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_hash, response_json, status_code, expires_at
                FROM idempotency_records
                WHERE tenant_id = ? AND endpoint = ? AND idem_key = ?
                """,
                (tenant_id, endpoint, idem_key),
            ).fetchone()
            if row is None:
                return None
            if row["expires_at"] <= now:
                connection.execute(
                    """
                    DELETE FROM idempotency_records
                    WHERE tenant_id = ? AND endpoint = ? AND idem_key = ?
                    """,
                    (tenant_id, endpoint, idem_key),
                )
                return None
            return {
                "request_hash": row["request_hash"],
                "response": json.loads(row["response_json"]),
                "status_code": row["status_code"],
            }

    @staticmethod
    def _insert_idempotency(
        connection: sqlite3.Connection,
        *,
        tenant_id: str,
        endpoint: str,
        idem_key: str,
        request_hash: str,
        response: dict[str, Any],
        status_code: int,
        created_at: str,
        expires_at: str,
    ) -> None:
        connection.execute(
            """
            INSERT INTO idempotency_records (
                tenant_id, endpoint, idem_key, request_hash, response_json,
                status_code, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                tenant_id,
                endpoint,
                idem_key,
                request_hash,
                json.dumps(response, sort_keys=True, separators=(",", ":")),
                status_code,
                created_at,
                expires_at,
            ),
        )

    def store_recommendation(
        self,
        *,
        tenant_id: str,
        idem_key: str,
        request_hash: str,
        request_json: dict[str, Any],
        response: dict[str, Any],
        idempotency_expires_at: str,
    ) -> None:
        metadata = response["metadata"]
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO recommendation_sets (
                    tenant_id, set_id, request_id, user_id, session_id,
                    generated_at, expires_at, request_json, route, degraded
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    response["recommendation_set_id"],
                    response["request_id"],
                    response["user_id"],
                    request_json["session_id"],
                    response["generated_at"],
                    response["expires_at"],
                    json.dumps(request_json, sort_keys=True, separators=(",", ":")),
                    metadata["route"],
                    int(metadata["degraded"]),
                ),
            )
            connection.executemany(
                """
                INSERT INTO recommendation_items (
                    tenant_id, set_id, item_id, position, title, reason,
                    tracking_token
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        tenant_id,
                        response["recommendation_set_id"],
                        item["item_id"],
                        item["rank"],
                        item["title"],
                        item["reason"],
                        item["tracking_token"],
                    )
                    for item in response["recommendations"]
                ],
            )
            self._insert_idempotency(
                connection,
                tenant_id=tenant_id,
                endpoint="recommendations",
                idem_key=idem_key,
                request_hash=request_hash,
                response=response,
                status_code=200,
                created_at=response["generated_at"],
                expires_at=idempotency_expires_at,
            )

    def get_recommendation_set(
        self, tenant_id: str, set_id: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT user_id, session_id, generated_at, expires_at
                FROM recommendation_sets
                WHERE tenant_id = ? AND set_id = ?
                """,
                (tenant_id, set_id),
            ).fetchone()
            return dict(row) if row else None

    def get_recommendation_item(
        self, tenant_id: str, set_id: str, item_id: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT item_id, position, tracking_token
                FROM recommendation_items
                WHERE tenant_id = ? AND set_id = ? AND item_id = ?
                """,
                (tenant_id, set_id, item_id),
            ).fetchone()
            return dict(row) if row else None

    def get_feedback(
        self, tenant_id: str, event_id: str
    ) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT request_hash, received_at
                FROM feedback_events
                WHERE tenant_id = ? AND event_id = ?
                """,
                (tenant_id, event_id),
            ).fetchone()
            return dict(row) if row else None

    def store_feedback(
        self,
        *,
        tenant_id: str,
        idem_key: str,
        request_hash: str,
        event: dict[str, Any] | None,
        response: dict[str, Any],
        created_at: str,
        idempotency_expires_at: str,
        status_code: int,
    ) -> None:
        with self._connect() as connection:
            if event is not None:
                connection.execute(
                    """
                    INSERT INTO feedback_events (
                        tenant_id, event_id, set_id, item_id, position, action,
                        session_id, client_timestamp, dwell_ms, request_hash,
                        request_json, received_at, schema_version
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
                    """,
                    (
                        tenant_id,
                        event["event_id"],
                        event["set_id"],
                        event["item_id"],
                        event["position"],
                        event["action"],
                        event["session_id"],
                        event["client_timestamp"],
                        event["dwell_ms"],
                        request_hash,
                        json.dumps(event, sort_keys=True, separators=(",", ":")),
                        created_at,
                    ),
                )
            self._insert_idempotency(
                connection,
                tenant_id=tenant_id,
                endpoint="feedback",
                idem_key=idem_key,
                request_hash=request_hash,
                response=response,
                status_code=status_code,
                created_at=created_at,
                expires_at=idempotency_expires_at,
            )

    def feedback_count(self, tenant_id: str) -> int:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM feedback_events WHERE tenant_id = ?",
                (tenant_id,),
            ).fetchone()
            return int(row["count"])
