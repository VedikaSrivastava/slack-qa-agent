"""Slack Bolt application factory."""

from __future__ import annotations

from typing import Any

import structlog
from slack_bolt.async_app import AsyncApp

from slack_qa_agent.config import Settings
from slack_qa_agent.execution.dispatcher import QuestionDispatcher
from slack_qa_agent.integrations.slack.parsing import parse_app_mention

logger = structlog.get_logger(__name__)


def create_slack_app(settings: Settings, dispatcher: QuestionDispatcher) -> AsyncApp:
    """Create a single-workspace Slack app with SDK-managed signature validation."""

    if not settings.slack_configured:
        raise RuntimeError("Slack credentials are required to mount the Slack Events API")

    assert settings.slack_bot_token is not None
    assert settings.slack_signing_secret is not None

    app = AsyncApp(
        token=settings.slack_bot_token.get_secret_value(),
        signing_secret=settings.slack_signing_secret.get_secret_value(),
        raise_error_for_unhandled_request=True,
    )

    @app.event("app_mention")
    async def handle_app_mention(
        body: dict[str, Any],
        event: dict[str, Any],
    ) -> None:
        if event.get("bot_id") or event.get("subtype") == "bot_message":
            return

        job = parse_app_mention(body, event)
        event_ids = await dispatcher.enqueue(job)
        logger.info(
            "slack_question_enqueued",
            slack_event_id=job.event_id,
            conversation_id=job.conversation_id,
            inngest_event_ids=event_ids,
        )

    return app
