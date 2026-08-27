"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

APPLICATION_VERSION = "0.1.0"
PROMPT_VERSION = "v1"
# v3 adds recall-safe structured filters to v2's cumulative, deduplicated refinement evidence.
RETRIEVAL_VERSION = "v3"
LANGSMITH_PROJECT_NAME = "slack-qa-agent"

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
    """Configuration required to execute the transport-independent grounded agent."""

    app_env: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    openai_api_key: SecretStr
    knowledge_db_path: Path = Path("data/synthetic_startup.sqlite")

    @field_validator("log_level", mode="before")
    @classmethod
    def normalize_log_level(cls, value: object) -> object:
        return value.upper() if isinstance(value, str) else value

    @field_validator("openai_api_key", mode="after")
    @classmethod
    def reject_blank_openai_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be blank")
        return value

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"


class LangSmithSettings(BaseSettings):
    """Credentials required for dataset-only LangSmith operations."""

    model_config = SETTINGS_CONFIG

    langsmith_api_key: SecretStr

    @field_validator("langsmith_api_key", mode="after")
    @classmethod
    def reject_blank_langsmith_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("LANGSMITH_API_KEY must not be blank")
        return value


class AugmentationSettings(LangSmithSettings):
    """Credentials required to generate and store evaluation candidates."""

    openai_api_key: SecretStr

    @field_validator("openai_api_key", mode="after")
    @classmethod
    def reject_blank_openai_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("OPENAI_API_KEY must not be blank")
        return value


class EvaluationSettings(AgentRuntimeSettings):
    """Strict settings for LangSmith-backed evaluation commands."""

    langsmith_api_key: SecretStr

    @field_validator("langsmith_api_key", mode="after")
    @classmethod
    def reject_blank_langsmith_key(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("LANGSMITH_API_KEY must not be blank")
        return value


class SlackApplicationSettings(AgentRuntimeSettings):
    """Validated Slack runtime settings; invalid configuration stops startup."""

    slack_bot_token: SecretStr
    slack_signing_secret: SecretStr
    inngest_dev: bool = True
    inngest_event_key: SecretStr | None = None
    inngest_signing_key: SecretStr | None = None
    langsmith_tracing: bool = False
    langsmith_api_key: SecretStr | None = None

    @field_validator("slack_bot_token", "slack_signing_secret", mode="after")
    @classmethod
    def reject_blank_slack_secrets(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required Slack configuration values must not be blank")
        return value

    @model_validator(mode="after")
    def validate_integration_contract(self) -> SlackApplicationSettings:
        if self.langsmith_tracing and (
            self.langsmith_api_key is None or not self.langsmith_api_key.get_secret_value().strip()
        ):
            raise ValueError("LANGSMITH_API_KEY is required when LANGSMITH_TRACING is true")
        if self.app_env == "production":
            if self.inngest_dev:
                raise ValueError("INNGEST_DEV must be false in production")
            if self.inngest_event_key is None or not self.inngest_event_key.get_secret_value():
                raise ValueError("INNGEST_EVENT_KEY is required in production")
            if self.inngest_signing_key is None or not self.inngest_signing_key.get_secret_value():
                raise ValueError("INNGEST_SIGNING_KEY is required in production")
        return self


@lru_cache(maxsize=1)
def get_slack_application_settings() -> SlackApplicationSettings:
    return SlackApplicationSettings()


@lru_cache(maxsize=1)
def get_agent_runtime_settings() -> AgentRuntimeSettings:
    return AgentRuntimeSettings()


@lru_cache(maxsize=1)
def get_database_settings() -> DatabaseSettings:
    return DatabaseSettings()
