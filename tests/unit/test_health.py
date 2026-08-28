import hashlib
import hmac
import json
import time
from collections.abc import Iterator
from pathlib import Path
from types import TracebackType
from typing import cast
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from slack_sdk.web.async_client import AsyncWebClient

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


@pytest.fixture(autouse=True)
def stub_slack_startup_authorization() -> Iterator[AsyncMock]:
    with patch(
        "knowledge_assistant.api.app.StartupSlackAuthorizer.initialize",
        new_callable=AsyncMock,
    ) as initialize:
        yield initialize


def _settings(tmp_path: Path) -> SlackApplicationSettings:
    return SlackApplicationSettings(
        _env_file=None,
        app_env="test",
        openai_api_key="test-key",
        slack_bot_token="xoxb-test",
        slack_signing_secret="test-signing-secret",
        slack_routing_policy="explicit_mentions_only",
        database_url="postgresql+asyncpg://user:password@127.0.0.1:1/test",
        knowledge_db_path=tmp_path / "missing.sqlite",
    )


def _slack_signature_headers(raw_body: str, signing_secret: str) -> dict[str, str]:
    timestamp = str(int(time.time()))
    signature_base = f"v0:{timestamp}:{raw_body}".encode()
    signature = hmac.new(signing_secret.encode(), signature_base, hashlib.sha256).hexdigest()
    return {
        "content-type": "application/json",
        "x-slack-request-timestamp": timestamp,
        "x-slack-signature": f"v0={signature}",
    }


def test_health_endpoint_reports_liveness_for_valid_configuration(
    tmp_path: Path,
    stub_slack_startup_authorization: AsyncMock,
) -> None:
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
    stub_slack_startup_authorization.assert_awaited_once_with()
    engine.dispose.assert_awaited_once()


def test_slack_publisher_disables_hidden_transport_retries(tmp_path: Path) -> None:
    engine = AsyncMock()
    slack_client = AsyncWebClient(token="xoxb-test", retry_handlers=[], timeout=20)
    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        patch(
            "knowledge_assistant.api.app.AsyncWebClient",
            return_value=slack_client,
        ) as slack_client_factory,
    ):
        create_app(_settings(tmp_path))

    slack_client_factory.assert_called_once_with(
        token="xoxb-test",
        retry_handlers=[],
        timeout=20,
    )


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
    assert response.json() == {
        "status": "not_ready",
        "knowledge_database": "missing",
        "postgres": "unavailable",
    }
    postgres_readiness.assert_awaited_once_with(engine)
    engine.dispose.assert_awaited_once()
    assert "/slack/events" in paths
    assert "/api/inngest" in paths
    assert "/api/ask" not in paths
    assert "/" not in paths


def test_readiness_reports_only_directly_probed_dependencies(tmp_path: Path) -> None:
    engine = AsyncMock()
    postgres_readiness = AsyncMock(return_value=True)
    (tmp_path / "missing.sqlite").touch()
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

    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "knowledge_database": "available",
        "postgres": "available",
    }
    postgres_readiness.assert_awaited_once_with(engine)
    engine.dispose.assert_awaited_once()


def test_startup_fails_when_knowledge_database_is_missing(tmp_path: Path) -> None:
    engine = AsyncMock()
    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        pytest.raises(FileNotFoundError, match="Knowledge database does not exist"),
        TestClient(create_app(_settings(tmp_path))),
    ):
        pass

    engine.dispose.assert_awaited_once()


def test_slack_events_accepts_correctly_signed_url_verification(tmp_path: Path) -> None:
    engine = AsyncMock()
    raw_body = json.dumps(
        {
            "token": "legacy-verification-token",
            "challenge": "verified-challenge",
            "type": "url_verification",
        },
        separators=(",", ":"),
    )
    headers = _slack_signature_headers(raw_body, "test-signing-secret")

    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        patch(
            "knowledge_assistant.api.app.create_question_processor",
            return_value=FakeProcessorContext(),
        ),
        TestClient(create_app(_settings(tmp_path))) as client,
    ):
        response = client.post("/slack/events", content=raw_body, headers=headers)

    assert response.status_code == 200
    assert response.json() == {"challenge": "verified-challenge"}
    engine.dispose.assert_awaited_once()


def test_slack_events_rejects_invalid_signature(tmp_path: Path) -> None:
    engine = AsyncMock()
    raw_body = json.dumps(
        {
            "token": "legacy-verification-token",
            "challenge": "must-not-be-returned",
            "type": "url_verification",
        },
        separators=(",", ":"),
    )
    headers = _slack_signature_headers(raw_body, "wrong-signing-secret")

    with (
        patch("knowledge_assistant.api.app.create_database_engine", return_value=engine),
        patch(
            "knowledge_assistant.api.app.create_question_processor",
            return_value=FakeProcessorContext(),
        ),
        TestClient(create_app(_settings(tmp_path))) as client,
    ):
        response = client.post("/slack/events", content=raw_body, headers=headers)

    assert response.status_code == 401
    assert response.json() == {"error": "invalid request"}
    engine.dispose.assert_awaited_once()
