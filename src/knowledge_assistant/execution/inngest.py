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


def _log_safe_error_publish_failure(job: QuestionJob, exc: Exception) -> None:
    """Record the failure class without exposing provider text or exception arguments."""

    logger.error(
        "slack_safe_error_publish_failed",
        agent_run_id=str(job.agent_run_id),
        conversation_id=job.conversation_id,
        error_code="slack_safe_error_publish_failed",
        exception_class=type(exc).__name__,
    )


def create_inngest_client(settings: SlackApplicationSettings) -> inngest.Inngest:
    return inngest.Inngest(app_id="slack-qa-agent", is_production=settings.is_production)


async def ensure_agent_result(
    *,
    job: QuestionJob,
    processor_provider: ProcessorProvider,
    ledger: RunLedger,
) -> AgentResponse:
    """Create or reuse the result persisted before the Inngest step is acknowledged."""

    persisted_response = await ledger.get_persisted_agent_result(job.agent_run_id)
    if persisted_response is not None:
        return persisted_response

    response = await processor_provider().answer(
        question=job.question,
        conversation_id=job.conversation_id,
        agent_run_id=str(job.agent_run_id),
    )
    # Commit the expensive output while status remains running. If Inngest loses the step
    # acknowledgement, the replay above can reuse it without marking delivery complete early.
    await ledger.persist_agent_result(job.agent_run_id, response)
    return response


def create_question_function(
    client: inngest.Inngest,
    *,
    processor_provider: ProcessorProvider,
    ledger: RunLedger,
    publisher: SlackPublisher,
) -> Any:
    """Register one durable function; LangGraph remains one coarse agent step."""

    # Inngest invokes this hook only after all configured function retries are exhausted.
    async def on_failure(ctx: inngest.Context) -> None:
        original = ctx.event.data.get("event")
        if not isinstance(original, dict) or not isinstance(original.get("data"), dict):
            logger.error(
                "inngest_failure_payload_invalid",
                error_code="inngest_failure_payload_invalid",
            )
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
            _log_safe_error_publish_failure(job, exc)

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
            response = await ensure_agent_result(
                job=job,
                processor_provider=processor_provider,
                ledger=ledger,
            )
            return response.model_dump(mode="json")

        async def publish_answer(payload: dict[str, Any]) -> str:
            return await publisher.publish_answer(
                job.agent_run_id, AgentResponse.model_validate(payload)
            )

        async def mark_completed(payload: dict[str, Any]) -> None:
            await ledger.mark_succeeded(job.agent_run_id, AgentResponse.model_validate(payload))

        # Separate memoized steps let retries resume after completed durable or user-visible effects.
        await ctx.step.run("mark-run-started", mark_started)
        await ctx.step.run("ensure-status-message", ensure_status_message)
        response_payload = cast(dict[str, Any], await ctx.step.run("run-agent", run_agent))
        response = AgentResponse.model_validate(response_payload)
        await ctx.step.run("publish-answer", publish_answer, response_payload)
        await ctx.step.run("mark-run-completed", mark_completed, response_payload)
        logger.info(
            "slack_question_completed",
            agent_run_id=str(job.agent_run_id),
            conversation_id=job.conversation_id,
            model_call_count=response.model_call_count,
            retrieval_round_count=response.retrieval_round_count,
            tool_call_count=response.tool_call_count,
        )
        return {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}

    return process_slack_question
