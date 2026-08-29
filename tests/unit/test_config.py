from pathlib import Path

import pytest
from pydantic import ValidationError

from knowledge_assistant.config import AgentRuntimeSettings, SlackApplicationSettings
from knowledge_assistant.integrations.slack.routing import SlackRoutingPolicy


def test_required_runtime_configuration_fails_fast(monkeypatch: pytest.MonkeyPatch) -> None:
    for variable_name in (
        "OPENAI_API_KEY",
        "SLACK_BOT_TOKEN",
        "SLACK_SIGNING_SECRET",
        "DATABASE_URL",
    ):
        monkeypatch.delenv(variable_name, raising=False)

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
        )


def test_database_url_requires_asyncpg_scheme() -> None:
    with pytest.raises(ValidationError, match="postgresql\\+asyncpg"):
        SlackApplicationSettings(
            _env_file=None,
            openai_api_key="test-key",
            slack_bot_token="xoxb-test",
            slack_signing_secret="test-signing-secret",
            database_url="postgresql://user:password@postgres/test",
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
        )


def test_agent_runtime_is_independent_from_slack() -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )

    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.langfuse_enabled is False


def test_langfuse_is_enabled_only_when_both_keys_are_present() -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
        langfuse_public_key="lf_pk_test",
        langfuse_secret_key="lf_sk_test",
    )

    assert settings.langfuse_enabled is True


def test_invalid_log_level_fails_instead_of_defaulting() -> None:
    with pytest.raises(ValidationError, match="log_level"):
        AgentRuntimeSettings(
            _env_file=None,
            openai_api_key="test-key",
            database_url="postgresql+asyncpg://user:password@postgres/test",
            log_level="verbose",
        )


def test_knowledge_database_path_has_a_repository_local_default() -> None:
    settings = AgentRuntimeSettings(
        _env_file=None,
        openai_api_key="test-key",
        database_url="postgresql+asyncpg://user:password@localhost/test",
    )

    assert settings.knowledge_db_path == Path("data/synthetic_startup.sqlite")


def test_slack_routing_policy_defaults_to_agent_owned_follow_ups() -> None:
    settings = SlackApplicationSettings(
        _env_file=None,
        openai_api_key="test-key",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-signing-secret",
        database_url="postgresql+asyncpg://user:password@localhost/test",
    )

    assert settings.slack_routing_policy is SlackRoutingPolicy.AGENT_OWNED_THREAD_FOLLOW_UPS
