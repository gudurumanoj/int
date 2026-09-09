from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True, slots=True)
class Settings:
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: float
    openai_max_retries: int
    openai_temperature: float
    openai_max_tokens: int
    db_path: Path
    workspace_root: Path
    docker_image: str
    approval_ttl_seconds: int
    sse_heartbeat_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            openai_timeout_seconds=_float_env("OPENAI_TIMEOUT_SECONDS", 60.0),
            openai_max_retries=_int_env("OPENAI_MAX_RETRIES", 2),
            openai_temperature=_float_env("OPENAI_TEMPERATURE", 0.0),
            openai_max_tokens=_int_env("OPENAI_MAX_TOKENS", 2048),
            db_path=Path(os.getenv("CODE_AGENT_DB_PATH", "./code_agent.db")),
            workspace_root=Path(
                os.getenv("CODE_AGENT_WORKSPACE_ROOT", "./workspaces")
            ),
            docker_image=os.getenv(
                "CODE_AGENT_DOCKER_IMAGE", "python:3.11-slim"
            ),
            approval_ttl_seconds=_int_env(
                "CODE_AGENT_APPROVAL_TTL_SECONDS", 300
            ),
            sse_heartbeat_seconds=_float_env(
                "CODE_AGENT_SSE_HEARTBEAT_SECONDS", 15.0
            ),
        )
