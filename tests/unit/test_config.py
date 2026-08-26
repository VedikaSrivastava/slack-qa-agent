from pathlib import Path

import pytest
from pydantic import ValidationError

from slack_qa_agent.config import ExperimentSettings, Settings


def test_required_runtime_configuration_fails_fast() -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_blank_secret_values_fail_fast() -> None:
    with pytest.raises(ValidationError):
        Settings(
            _env_file=None,
            openai_api_key="",
            slack_bot_token="",
            slack_signing_secret="",
            database_url="",
            knowledge_db_path=Path("data/example.sqlite"),
        )


def test_database_url_requires_asyncpg_scheme() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        Settings(
            _env_file=None,
            openai_api_key="test-key",
            slack_bot_token="xoxb-test",
            slack_signing_secret="test-signing-secret",
            database_url="postgresql://user:password@postgres/test",
            knowledge_db_path=Path("data/example.sqlite"),
        )


def test_production_requires_inngest_credentials() -> None:
    with pytest.raises(ValidationError, match="INNGEST_DEV"):
        Settings(
            _env_file=None,
            app_env="production",
            openai_api_key="test-key",
            slack_bot_token="xoxb-test",
            slack_signing_secret="test-signing-secret",
            database_url="postgresql+asyncpg://user:password@postgres/test",
            knowledge_db_path=Path("data/example.sqlite"),
            langsmith_tracing=True,
            langsmith_api_key="test-langsmith-key",
            langsmith_project="slack-qa-agent",
        )


def test_experiment_configuration_is_independent_from_slack() -> None:
    settings = ExperimentSettings(
        _env_file=None,
        app_env="test",
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
        knowledge_db_path=Path("data/example.sqlite"),
        langsmith_tracing=True,
        langsmith_api_key="test-langsmith-key",
        langsmith_project="slack-qa-agent",
    )

    assert settings.langsmith_project == "slack-qa-agent"
