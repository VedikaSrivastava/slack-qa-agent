from pathlib import Path

from fastapi.testclient import TestClient

from slack_qa_agent.api.app import create_app
from slack_qa_agent.config import Settings


def test_health_endpoint_does_not_require_runtime_secrets(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        knowledge_db_path=tmp_path / "missing.sqlite",
    )
    client = TestClient(create_app(settings))

    response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_readiness_reports_missing_database(tmp_path: Path) -> None:
    settings = Settings(
        _env_file=None,
        app_env="test",
        knowledge_db_path=tmp_path / "missing.sqlite",
    )
    client = TestClient(create_app(settings))

    response = client.get("/readyz")

    assert response.status_code == 503
    assert response.json()["database"] == "missing"
