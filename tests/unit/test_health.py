from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from knowledge_assistant.api.app import create_app
from knowledge_assistant.application.question_processor import QuestionProcessor
from knowledge_assistant.config import SlackApplicationSettings


class FakeProcessorContext:
    async def __aenter__(self) -> QuestionProcessor:
        return cast(QuestionProcessor, object())

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


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
    engine = AsyncMock()
    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        patch(
            "knowledge_assistant.api.app.create_question_processor",
            return_value=FakeProcessorContext(),
        ),
        TestClient(create_app(_settings(tmp_path))) as client,
    ):
        response = client.get("/healthz")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    engine.dispose.assert_awaited_once()


def test_readiness_reports_missing_database(tmp_path: Path) -> None:
    engine = AsyncMock()
    postgres_readiness = AsyncMock(return_value=False)
    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        patch(
            "knowledge_assistant.api.app.create_question_processor",
            return_value=FakeProcessorContext(),
        ),
        patch("knowledge_assistant.api.app.database_is_ready", postgres_readiness),
        TestClient(create_app(_settings(tmp_path))) as client,
    ):
        response = client.get("/readyz")
        paths = {getattr(route, "path", None) for route in cast(FastAPI, client.app).routes}

    assert response.status_code == 503
    assert response.json()["knowledge_database"] == "missing"
    assert response.json()["postgres"] == "unavailable"
    postgres_readiness.assert_awaited_once_with(engine)
    engine.dispose.assert_awaited_once()
    assert "/slack/events" in paths
    assert "/api/inngest" in paths
    assert "/api/ask" not in paths
    assert "/" not in paths


def test_startup_fails_when_knowledge_database_is_missing(tmp_path: Path) -> None:
    engine = AsyncMock()
    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        pytest.raises(FileNotFoundError, match="Knowledge database does not exist"),
        TestClient(create_app(_settings(tmp_path))),
    ):
        pass

    engine.dispose.assert_awaited_once()
