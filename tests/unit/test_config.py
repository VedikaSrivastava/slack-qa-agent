from pathlib import Path

import pytest
from pydantic import ValidationError

from knowledge_assistant.config import (
    LANGSMITH_PROJECT_NAME,
    AgentRuntimeSettings,
    EvaluationSettings,
    SlackApplicationSettings,
)


def test_required_runtime_configuration_fails_fast() -> None:
    with pytest.raises(ValidationError):
        SlackApplicationSettings(_env_file=None)


def test_blank_secret_values_fail_fast() -> None:
    with pytest.raises(ValidationError):
        SlackApplicationSettings(
            _env_file=None,
            openai_api_key="",
            slack_bot_token="",
            slack_signing_secret="",
            database_url="",
            langsmith_api_key="",
        )


def test_database_url_requires_asyncpg_scheme() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        SlackApplicationSettings(
            _env_file=None,
            openai_api_key="test-key",
            slack_bot_token="xoxb-test",
            slack_signing_secret="test-signing-secret",
            database_url="postgresql://user:password@postgres/test",
            langsmith_api_key="test-langsmith-key",
        )


def test_production_requires_inngest_credentials() -> None:
    with pytest.raises(ValidationError, match="INNGEST_DEV"):
        SlackApplicationSettings(
            _env_file=None,
            app_env="production",
            openai_api_key="test-key",
            slack_bot_token="xoxb-test",
            slack_signing_secret="test-signing-secret",
            database_url="postgresql+asyncpg://user:password@postgres/test",
            knowledge_db_path=Path("data/example.sqlite"),
            langsmith_api_key="test-langsmith-key",
        )


def test_experiment_configuration_is_independent_from_slack() -> None:
    settings = EvaluationSettings(
        _env_file=None,
        app_env="test",
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
        knowledge_db_path=Path("data/example.sqlite"),
        langsmith_api_key="test-langsmith-key",
    )

    assert LANGSMITH_PROJECT_NAME == "slack-qa-agent"
    assert settings.knowledge_db_path == Path("data/example.sqlite")


def test_agent_runtime_is_independent_from_slack_and_langsmith() -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )

    assert settings.openai_api_key.get_secret_value() == "test-key"


def test_slack_tracing_requires_langsmith_key_when_enabled() -> None:
    with pytest.raises(ValidationError, match="LANGSMITH_API_KEY"):
        SlackApplicationSettings(
            _env_file=None,
            openai_api_key="test-key",
            slack_bot_token="xoxb-test",
            slack_signing_secret="test-signing-secret",
            database_url="postgresql+asyncpg://user:password@postgres/test",
            langsmith_tracing=True,
        )


def test_invalid_log_level_fails_instead_of_defaulting() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        AgentRuntimeSettings(
            _env_file=None,
            openai_api_key="test-key",
            database_url="postgresql+asyncpg://user:password@postgres/test",
            log_level="verbose",
        )


def test_knowledge_database_path_has_a_repository_local_default() -> None:
    settings = EvaluationSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@localhost/test",
        langsmith_api_key="test-langsmith-key",
    )

    assert settings.knowledge_db_path == Path("data/synthetic_startup.sqlite")
