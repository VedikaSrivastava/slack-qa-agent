"""Inngest client and coarse-grained durable Slack question workflow."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, cast

import inngest
import structlog

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.application.question_processor import QuestionProcessor
from knowledge_assistant.config import SlackApplicationSettings
from knowledge_assistant.execution.dispatcher import QUESTION_RECEIVED_EVENT
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.integrations.slack.publisher import SlackPublisher
from knowledge_assistant.persistence.repositories import RunLedger

logger = structlog.get_logger(__name__)
ProcessorProvider = Callable[[], QuestionProcessor]


def create_inngest_client(settings: SlackApplicationSettings) -> inngest.Inngest:
    return inngest.Inngest(app_id="slack-qa-agent", is_production=settings.is_production)


def create_question_function(
    client: inngest.Inngest,
    *,
    processor_provider: ProcessorProvider,
    ledger: RunLedger,
    publisher: SlackPublisher,
) -> Any:
    """Register one durable function; LangGraph remains one coarse agent step."""

    async def on_failure(ctx: inngest.Context) -> None:
        original = ctx.event.data.get("event")
        if not isinstance(original, dict) or not isinstance(original.get("data"), dict):
            logger.error("inngest_failure_payload_invalid")
            return
        job = QuestionJob.model_validate(original["data"])
        await ledger.mark_failed(
            job.agent_run_id,
            code="inngest_retries_exhausted",
            message="Background processing failed after retry exhaustion.",
        )
        try:
            await publisher.publish_safe_error(job.agent_run_id)
        except Exception as exc:
            logger.error(
                "slack_safe_error_publish_failed",
                agent_run_id=str(job.agent_run_id),
                exception_class=type(exc).__name__,
            )

    @client.create_function(
        fn_id="process-slack-question",
        name="Process Slack question",
        trigger=inngest.TriggerEvent(event=QUESTION_RECEIVED_EVENT),
        retries=3,
        on_failure=on_failure,
        idempotency="event.data.event_id",
        concurrency=[
            inngest.Concurrency(limit=1, key="event.data.conversation_id"),
            inngest.Concurrency(limit=8, key='"openai"', scope="env"),
        ],
    )
    async def process_slack_question(ctx: inngest.Context) -> dict[str, Any]:
        job = QuestionJob.model_validate(ctx.event.data)

        async def mark_started() -> None:
            await ledger.mark_running(job.agent_run_id)

        async def ensure_status_message() -> str:
            return await publisher.ensure_placeholder(job.agent_run_id)

        async def run_agent() -> dict[str, Any]:
            completed = await ledger.get_completed_result(job.agent_run_id)
            if completed is not None:
                return completed.model_dump(mode="json")
            response = await processor_provider().answer(
                question=job.question,
                conversation_id=job.conversation_id,
                agent_run_id=str(job.agent_run_id),
            )
            return response.model_dump(mode="json")

        async def publish_answer(payload: dict[str, Any]) -> str:
            return await publisher.publish_answer(
                job.agent_run_id, AgentResponse.model_validate(payload)
            )

        async def mark_completed(payload: dict[str, Any]) -> None:
            await ledger.mark_succeeded(job.agent_run_id, AgentResponse.model_validate(payload))

        await ctx.step.run("mark-run-started", mark_started)
        await ctx.step.run("ensure-status-message", ensure_status_message)
        response_payload = cast(dict[str, Any], await ctx.step.run("run-agent", run_agent))
        await ctx.step.run("publish-answer", publish_answer, response_payload)
        await ctx.step.run("mark-run-completed", mark_completed, response_payload)
        logger.info(
            "slack_question_completed",
            agent_run_id=str(job.agent_run_id),
            conversation_id=job.conversation_id,
        )
        return {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}

    return process_slack_question
