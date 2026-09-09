from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def payload_hash(value: Any) -> str:
    return sha256_bytes(canonical_json(value).encode("utf-8"))
