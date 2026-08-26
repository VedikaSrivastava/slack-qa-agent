"""Slack Bolt application factory."""

from __future__ import annotations

from typing import Any

import structlog
from pydantic import ValidationError
from slack_bolt.async_app import AsyncApp

from slack_qa_agent.config import Settings
from slack_qa_agent.execution.dispatcher import QuestionDispatcher
from slack_qa_agent.integrations.slack.parsing import parse_app_mention
from slack_qa_agent.persistence.repositories import RunLedger

logger = structlog.get_logger(__name__)


def create_slack_app(
    settings: Settings,
    dispatcher: QuestionDispatcher,
    ledger: RunLedger,
) -> AsyncApp:
    """Create a single-workspace Slack app with SDK-managed signature validation."""

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

        try:
            job = parse_app_mention(body, event)
        except (KeyError, ValidationError) as exc:
            logger.warning(
                "slack_event_rejected",
                error_code="malformed_app_mention",
                exception_class=type(exc).__name__,
            )
            return
        run_id, created = await ledger.create_queued(job)
        job = job.model_copy(update={"agent_run_id": run_id})
        event_ids = await dispatcher.enqueue(job)
        if event_ids:
            await ledger.attach_inngest_event(run_id, event_ids[0])
        logger.info(
            "slack_question_enqueued",
            agent_run_id=str(run_id),
            slack_event_id=job.event_id,
            conversation_id=job.conversation_id,
            inngest_event_ids=event_ids,
            new_run=created,
        )

    return app
