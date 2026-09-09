from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Settings:
    """Configuration read from environment variables at app construction."""

    data_dir: Path
    openai_base_url: str
    openai_api_key: str
    openai_model: str
    openai_timeout_seconds: float
    openai_max_retries: int
    openai_temperature: float
    openai_max_tokens: int
    docker_image: str
    docker_cpus: str
    docker_memory: str
    docker_timeout_seconds: float

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            data_dir=Path(os.getenv("DATASCI_DATA_DIR", "./data")).resolve(),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", "not-configured"),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
            openai_timeout_seconds=float(
                os.getenv("OPENAI_TIMEOUT_SECONDS", "60")
            ),
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "2")),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0")),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "2000")),
            docker_image=os.getenv(
                "DATASCI_DOCKER_IMAGE", "datasci-sandbox:py311"
            ),
            docker_cpus=os.getenv("DATASCI_DOCKER_CPUS", "1"),
            docker_memory=os.getenv("DATASCI_DOCKER_MEMORY", "1g"),
            docker_timeout_seconds=float(
                os.getenv("DATASCI_DOCKER_TIMEOUT_SECONDS", "120")
            ),
        )
