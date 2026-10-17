"""Application settings, loaded from environment variables / a local .env file."""

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Typed settings. Missing required values raise a validation error at startup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    google_api_key: SecretStr
    llm_model: str  # intentionally no default: model IDs get retired
    llm_timeout_s: float = 30.0
    llm_max_retries: int = 2
    log_level: str = "INFO"
    checkpoint_db_path: Path = Path("data/checkpoints.db")
    uploads_dir: Path = Path("data/uploads")


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance (parsed once per process)."""
    return Settings()