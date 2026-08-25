"""Inngest client and function registration.

Inngest owns queueing, retries, idempotency, and execution observability. LangGraph
continues to own the internal reasoning workflow and conversational checkpoints.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from typing import Any

import inngest

from slack_qa_agent.config import Settings
from slack_qa_agent.execution.dispatcher import QUESTION_RECEIVED_EVENT
from slack_qa_agent.execution.models import QuestionJob

JobHandler = Callable[[QuestionJob], Awaitable[dict[str, Any]]]


def create_inngest_client(settings: Settings) -> inngest.Inngest:
    return inngest.Inngest(
        app_id="slack-qa-agent",
        is_production=settings.is_production,
        logger=logging.getLogger("slack_qa_agent.inngest"),
    )


def create_question_function(
    client: inngest.Inngest,
    handler: JobHandler,
) -> Any:
    """Register the durable worker around an injected application handler."""

    @client.create_function(
        fn_id="answer-slack-question",
        name="Answer Slack question",
        trigger=inngest.TriggerEvent(event=QUESTION_RECEIVED_EVENT),
        retries=3,
        idempotency="event.data.event_id",
    )
    async def answer_slack_question(ctx: inngest.Context) -> dict[str, Any]:
        job = QuestionJob.model_validate(ctx.event.data)
        return await ctx.step.run("run-question-processor", handler, job)

    return answer_slack_question
