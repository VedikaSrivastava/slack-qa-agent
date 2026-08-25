from pathlib import Path

from slack_qa_agent.config import Settings


def test_settings_are_safe_without_secrets() -> None:
    settings = Settings(
        _env_file=None,
        knowledge_db_path=Path("data/example.sqlite"),
    )

    assert settings.slack_configured is False
    assert settings.agent_configured is False
    assert settings.is_production is False
