"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Secrets remain optional so health checks can run before setup."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    openai_api_key: SecretStr | None = None
    openai_model: str | None = None

    slack_bot_token: SecretStr | None = None
    slack_signing_secret: SecretStr | None = None

    knowledge_db_path: Path = Path("data/synthetic_startup.sqlite")
    checkpoint_db_path: Path = Path(".runtime/checkpoints.sqlite")

    @property
    def slack_configured(self) -> bool:
        return self.slack_bot_token is not None and self.slack_signing_secret is not None

    @property
    def agent_configured(self) -> bool:
        return self.openai_api_key is not None and bool(self.openai_model)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
