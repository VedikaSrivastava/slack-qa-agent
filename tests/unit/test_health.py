from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from knowledge_assistant.api.app import create_app
from knowledge_assistant.config import SlackApplicationSettings


def _settings(tmp_path: Path) -> SlackApplicationSettings:
    return SlackApplicationSettings(
        _env_file=None,
        app_env="test",
        openai_api_key="test-key",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-signing-secret",
        database_url="postgresql+asyncpg://user:password@127.0.0.1:1/test",
        knowledge_db_path=tmp_path / "missing.sqlite",
    )


def test_health_endpoint_reports_liveness_for_valid_configuration(tmp_path: Path) -> None:
    client = TestClient(create_app(_settings(tmp_path)))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_missing_database(tmp_path: Path) -> None:
    app = create_app(_settings(tmp_path))
    client = TestClient(app)

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["knowledge_database"] == "missing"
    paths = {getattr(route, "path", None) for route in app.routes}
    assert "/slack/events" in paths
    assert "/api/inngest" in paths
    assert "/api/ask" not in paths
    assert "/" not in paths


def test_startup_fails_when_knowledge_database_is_missing(tmp_path: Path) -> None:
    with (
        pytest.raises(FileNotFoundError, match="Knowledge database does not exist"),
        TestClient(create_app(_settings(tmp_path))),
    ):
        pass
