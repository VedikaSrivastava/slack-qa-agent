"""Slack Bolt application factory."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import structlog
from pydantic import ValidationError
from slack_bolt.async_app import AsyncApp

from knowledge_assistant.config import SlackApplicationSettings
from knowledge_assistant.execution.dispatcher import QuestionDispatcher
from knowledge_assistant.integrations.slack.parsing import parse_app_mention, strip_bot_mentions
from knowledge_assistant.persistence.repositories import RunLedger

logger = structlog.get_logger(__name__)
SlackResponder = Callable[..., Awaitable[Any]]
SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS = 2.0


async def process_app_mention(
    *,
    body: dict[str, Any],
    event: dict[str, Any],
    respond: SlackResponder,
    dispatcher: QuestionDispatcher,
    ledger: RunLedger,
) -> None:
    """Validate one mention, persist it, and dispatch it for durable processing."""

    if event.get("bot_id") or event.get("subtype") is not None:
        return
    if not strip_bot_mentions(str(event.get("text", ""))):
        await respond(
            text="Please include a question after mentioning me.",
            thread_ts=str(event.get("thread_ts") or event.get("ts", "")),
        )
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
    # Bound the durable handoff below Slack's acknowledgement deadline. If any step times out,
    # Slack retries the same event ID and the idempotent ledger/dispatcher resume safely.
    async with asyncio.timeout(SLACK_DURABLE_HANDOFF_TIMEOUT_SECONDS):
        run_id, is_new_run = await ledger.create_queued(job)
        job = job.model_copy(update={"agent_run_id": run_id})
        event_ids = await dispatcher.enqueue(job)
        if not event_ids:
            raise RuntimeError("Inngest returned no event ID for the Slack question")
        await ledger.attach_inngest_event(run_id, event_ids[0])
    logger.info(
        "slack_question_enqueued",
        agent_run_id=str(run_id),
        slack_event_id=job.event_id,
        conversation_id=job.conversation_id,
        inngest_event_ids=event_ids,
        is_new_run=is_new_run,
    )


def create_slack_app(
    settings: SlackApplicationSettings,
    dispatcher: QuestionDispatcher,
    ledger: RunLedger,
) -> AsyncApp:
    """Create a single-workspace Slack app with SDK-managed signature validation."""

    app = AsyncApp(
        token=settings.slack_bot_token.get_secret_value(),
        signing_secret=settings.slack_signing_secret.get_secret_value(),
        raise_error_for_unhandled_request=True,
        # Slack must not receive a 2xx acknowledgment until the durable Inngest handoff succeeds.
        process_before_response=True,
    )

    @app.event("app_mention")
    async def handle_app_mention(
        body: dict[str, Any],
        event: dict[str, Any],
        say: SlackResponder,
    ) -> None:
        await process_app_mention(
            body=body,
            event=event,
            respond=say,
            dispatcher=dispatcher,
            ledger=ledger,
        )

    return app
