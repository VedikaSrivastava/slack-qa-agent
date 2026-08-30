"""Application configuration loaded from environment variables."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from knowledge_assistant.integrations.slack.routing import SlackRoutingPolicy

APPLICATION_VERSION = "0.1.0"
PROMPT_VERSION = "v18"
# v11 exposes shortlist provenance to grading and forces a full-evidence comparison refinement.
# Ordinary unknown-entity lookup and structured cohort retrieval retain their established paths.
RETRIEVAL_VERSION = "v11"
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
    langfuse_base_url: AnyHttpUrl = AnyHttpUrl("http://langfuse-web:3000")
    langfuse_public_key: SecretStr | None = None
    langfuse_secret_key: SecretStr | None = None

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

    @model_validator(mode="after")
    def validate_langfuse_credentials(self) -> AgentRuntimeSettings:
        public_key = (
            self.langfuse_public_key.get_secret_value().strip()
            if self.langfuse_public_key is not None
            else ""
        )
        secret_key = (
            self.langfuse_secret_key.get_secret_value().strip()
            if self.langfuse_secret_key is not None
            else ""
        )
        if bool(public_key) is not bool(secret_key):
            raise ValueError(
                "LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY must be configured together"
            )
        return self

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def langfuse_enabled(self) -> bool:
        return bool(
            self.langfuse_public_key
            and self.langfuse_public_key.get_secret_value().strip()
            and self.langfuse_secret_key
            and self.langfuse_secret_key.get_secret_value().strip()
        )


class SlackApplicationSettings(AgentRuntimeSettings):
    """Validated Slack runtime settings; invalid configuration stops startup."""

    slack_bot_token: SecretStr
    slack_signing_secret: SecretStr
    slack_routing_policy: SlackRoutingPolicy = SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS
    inngest_dev: bool = True
    inngest_event_key: SecretStr | None = None
    inngest_signing_key: SecretStr | None = None

    @field_validator("slack_bot_token", "slack_signing_secret", mode="after")
    @classmethod
    def reject_blank_slack_secrets(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("required Slack configuration values must not be blank")
        return value

    @model_validator(mode="after")
    def validate_integration_contract(self) -> SlackApplicationSettings:
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
