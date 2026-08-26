"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

SETTINGS_CONFIG = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8",
    extra="ignore",
    case_sensitive=False,
)


class DatabaseSettings(BaseSettings):
    """Database-only configuration used by migrations and checkpoint setup."""

    model_config = SETTINGS_CONFIG

    database_url: SecretStr

    @field_validator("database_url", mode="after")
    @classmethod
    def validate_database_url(cls, value: SecretStr) -> SecretStr:
        raw_value = value.get_secret_value()
        if not raw_value.strip():
            raise ValueError("DATABASE_URL must not be blank")
        if not raw_value.startswith("postgresql+asyncpg://"):
            raise ValueError("DATABASE_URL must use the postgresql+asyncpg:// scheme")
        return value

    def sqlalchemy_database_url(self) -> str:
        return self.database_url.get_secret_value()

    def psycopg_database_url(self) -> str:
        """Return a psycopg-compatible URL for LangGraph checkpoint storage."""

        return self.sqlalchemy_database_url().replace("postgresql+asyncpg://", "postgresql://", 1)


class AgentRuntimeSettings(DatabaseSettings):
    """Configuration required to execute and trace the grounded agent."""

    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    app_version: str = "dev"
    prompt_version: str = "v1"
    retrieval_version: str = "v1"

    openai_api_key: SecretStr
    knowledge_db_path: Path

    inngest_dev: bool = True
    inngest_base_url: str | None = None
    inngest_event_key: SecretStr | None = None
    inngest_signing_key: SecretStr | None = None

    langsmith_tracing: Literal[True]
    langsmith_api_key: SecretStr
    langsmith_project: Literal["slack-qa-agent"]

    @field_validator(
        "openai_api_key",
        "langsmith_api_key",
        mode="after",
    )
    @classmethod
    def reject_blank_secrets(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required configuration values must not be blank")
        return value

    @model_validator(mode="after")
    def validate_environment_contract(self) -> AgentRuntimeSettings:
        if self.app_env == "production":
            if self.inngest_dev:
                raise ValueError("INNGEST_DEV must be false in production")
            if self.inngest_event_key is None or not self.inngest_event_key.get_secret_value():
                raise ValueError("INNGEST_EVENT_KEY is required in production")
            if self.inngest_signing_key is None or not self.inngest_signing_key.get_secret_value():
                raise ValueError("INNGEST_SIGNING_KEY is required in production")
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


class EvaluationSettings(AgentRuntimeSettings):
    """Strict settings for local and LangSmith evaluation runs."""


class SlackApplicationSettings(AgentRuntimeSettings):
    """Validated Slack runtime settings; invalid configuration stops startup."""

    slack_bot_token: SecretStr
    slack_signing_secret: SecretStr

    @field_validator("slack_bot_token", "slack_signing_secret", mode="after")
    @classmethod
    def reject_blank_slack_secrets(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required Slack configuration values must not be blank")
        return value


@lru_cache(maxsize=1)
def get_slack_application_settings() -> SlackApplicationSettings:
    return SlackApplicationSettings()


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()
