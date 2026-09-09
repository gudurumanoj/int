from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv


PACKAGE_DIR = Path(__file__).resolve().parent


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_categories() -> tuple[str, ...]:
    value = os.getenv("ELIGIBILITY_ALLOWED_CATEGORIES", "")
    return tuple(part.strip().lower() for part in value.split(",") if part.strip())


@dataclass(frozen=True, slots=True)
class Settings:
    database_path: Path = Path("recsys.db")
    catalog_path: Path = field(
        default_factory=lambda: PACKAGE_DIR / "data" / "items.json"
    )
    default_tenant_id: str = "demo"
    hmac_secret: str = "unsafe-development-secret-change-me"

    recommendation_ttl_seconds: int = 1_800
    idempotency_ttl_seconds: int = 86_400
    llm_shortlist_size: int = 5
    eligibility_max_candidates: int = 100
    eligibility_require_available: bool = True
    eligibility_allowed_categories: tuple[str, ...] = ()

    openai_base_url: str = "https://api.openai.com/v1"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_timeout_seconds: float = 2.0
    openai_max_retries: int = 0
    openai_temperature: float = 0.0
    openai_max_tokens: int = 700

    @property
    def llm_enabled(self) -> bool:
        return bool(self.openai_api_key.strip())

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv()
        return cls(
            database_path=Path(os.getenv("DATABASE_PATH", "recsys.db")),
            catalog_path=Path(
                os.getenv("CATALOG_PATH", str(PACKAGE_DIR / "data" / "items.json"))
            ),
            default_tenant_id=os.getenv("DEFAULT_TENANT_ID", "demo"),
            hmac_secret=os.getenv(
                "HMAC_SECRET", "unsafe-development-secret-change-me"
            ),
            recommendation_ttl_seconds=int(
                os.getenv("RECOMMENDATION_TTL_SECONDS", "1800")
            ),
            idempotency_ttl_seconds=int(
                os.getenv("IDEMPOTENCY_TTL_SECONDS", "86400")
            ),
            llm_shortlist_size=int(os.getenv("LLM_SHORTLIST_SIZE", "5")),
            eligibility_max_candidates=int(
                os.getenv("ELIGIBILITY_MAX_CANDIDATES", "100")
            ),
            eligibility_require_available=_env_bool(
                "ELIGIBILITY_REQUIRE_AVAILABLE", True
            ),
            eligibility_allowed_categories=_env_categories(),
            openai_base_url=os.getenv(
                "OPENAI_BASE_URL", "https://api.openai.com/v1"
            ),
            openai_api_key=os.getenv("OPENAI_API_KEY", ""),
            openai_model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            openai_timeout_seconds=float(
                os.getenv("OPENAI_TIMEOUT_SECONDS", "2.0")
            ),
            openai_max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "0")),
            openai_temperature=float(os.getenv("OPENAI_TEMPERATURE", "0.0")),
            openai_max_tokens=int(os.getenv("OPENAI_MAX_TOKENS", "700")),
        )
