from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_DIR = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="ORCH_",
        extra="ignore",
    )

    sqlite_path: Path = PROJECT_DIR / "orchestrator.db"
    model_registry_path: Path = PROJECT_DIR / "models.json"
    idempotency_ttl_seconds: int = Field(default=86_400, gt=0)
    event_retention_seconds: int = Field(default=3_600, gt=0)
    default_timeout_ms: int = Field(default=5_000, gt=0)
    default_retries: int = Field(default=1, ge=0, le=5)
    default_max_tokens: int = Field(default=512, gt=0)
    default_temperature: float = Field(default=0.2, ge=0, le=2)
    default_api_key_env: str = "OPENAI_API_KEY"
