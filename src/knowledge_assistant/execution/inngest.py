"""Inngest functions for durable routing, agent work, delivery, and cleanup."""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import timedelta
from typing import Any, cast

import inngest
import structlog

from knowledge_assistant.agent.models import AgentResponse, FinalAnswerEvent, ProgressEvent
from knowledge_assistant.application.question_processor import StreamingQuestionProcessor
from knowledge_assistant.config import SlackApplicationSettings
from knowledge_assistant.execution.dispatcher import (
    FOLLOW_UP_CANDIDATE_EVENT,
    QUESTION_CANCELLED_EVENT,
    QUESTION_READY_EVENT,
    QUESTION_RECEIVED_EVENT,
    RESPONDER_CLASSIFICATION_EVENT,
)
from knowledge_assistant.execution.models import (
    FollowUpCandidateJob,
    QuestionCancellationJob,
    QuestionJob,
)
from knowledge_assistant.integrations.slack.publisher import (
    PreparedDelivery,
    ProgressSurfaceAction,
    ProgressSurfaceClaim,
    SlackDeliveryRejectedError,
    SlackPublisher,
)
from knowledge_assistant.integrations.slack.routing import (
    ResponderClassification,
    ResponderClassificationRequest,
    ResponderClassifier,
    SlackThreadIdentity,
    decide_responder_classification,
)
from knowledge_assistant.persistence.models import (
    DeliveryStatus,
    RunStatus,
    SlackTurnKind,
    SlackTurnStatus,
)
from knowledge_assistant.persistence.repositories import RunLedger

logger = structlog.get_logger(__name__)

INNGEST_APP_ID = "slack-qa-agent"
QUESTION_FUNCTION_ID = "process-slack-question"
PROGRESS_FUNCTION_ID = "initialize-slack-progress"
TURN_ROUTER_FUNCTION_ID = "route-slack-turn"
RESPONDER_CLASSIFIER_FUNCTION_ID = "classify-slack-follow-up"
QUESTION_FULLY_QUALIFIED_FUNCTION_ID = f"{INNGEST_APP_ID}-{QUESTION_FUNCTION_ID}"
TURN_ROUTER_FULLY_QUALIFIED_FUNCTION_ID = f"{INNGEST_APP_ID}-{TURN_ROUTER_FUNCTION_ID}"
FAILURE_CLEANUP_FUNCTION_ID = "cleanup-failed-slack-question"
CANCELLATION_CLEANUP_FUNCTION_ID = "cleanup-cancelled-slack-question"
TURN_FAILURE_CLEANUP_FUNCTION_ID = "cleanup-failed-slack-turn"
FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID = f"{INNGEST_APP_ID}-{FAILURE_CLEANUP_FUNCTION_ID}"
CANCELLATION_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID = (
    f"{INNGEST_APP_ID}-{CANCELLATION_CLEANUP_FUNCTION_ID}"
)
TURN_FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID = (
    f"{INNGEST_APP_ID}-{TURN_FAILURE_CLEANUP_FUNCTION_ID}"
)
FUNCTION_FAILED_EVENT = "inngest/function.failed"
FUNCTION_CANCELLED_EVENT = "inngest/function.cancelled"
TURN_WAIT_ATTEMPTS = 240
TURN_WAIT_INTERVAL = timedelta(seconds=15)
ProcessorProvider = Callable[[], StreamingQuestionProcessor]


def _log_safe_error_publish_failure(job: QuestionJob, exc: Exception) -> None:
    """Log only the provider exception class; Slack error text can contain sensitive data."""

    logger.error(
        "slack_safe_error_publish_failed",
        agent_run_id=str(job.agent_run_id),
        conversation_id=job.conversation_id,
        error_code="slack_safe_error_publish_failed",
        exception_class=type(exc).__name__,
    )


def create_inngest_client(settings: SlackApplicationSettings) -> inngest.Inngest:
    return inngest.Inngest(app_id=INNGEST_APP_ID, is_production=settings.is_production)


async def _ensure_progress_surface(
    ctx: inngest.Context,
    *,
    run_id: uuid.UUID,
    publisher: SlackPublisher,
) -> str | None:
    """Sequence stream claim, remote start, and acknowledgement as durable steps."""

    async def claim_surface() -> dict[str, Any]:
        # This is a cheap idempotent ownership claim. If another pipeline owns setup,
        # the same run_id should not fan out into duplicate stream-open calls.
        claim = await publisher.claim_progress_surface(run_id)
        return claim.model_dump(mode="json")

    claim_payload = cast(
        dict[str, Any],
        await ctx.step.run("claim-progress-surface", claim_surface),
    )
    claim = ProgressSurfaceClaim.model_validate(claim_payload)
    if claim.action is ProgressSurfaceAction.READY:
        return claim.timestamp
    if claim.action in {ProgressSurfaceAction.WAIT, ProgressSurfaceAction.DEGRADED}:
        # WAIT is a normal backoff path; DEGRADED is explicit fallback when stream mode
        # is not available but full answer execution can still proceed.
        return None

    timestamp_payload = await ctx.step.run(
        "start-slack-stream",
        lambda: publisher.start_claimed_stream(run_id),
    )
    timestamp = str(timestamp_payload) if timestamp_payload is not None else None
    result = await ctx.step.run(
        "acknowledge-slack-stream",
        lambda: publisher.finish_progress_surface(run_id, timestamp),
    )
    return str(result) if result is not None else None


async def _stream_agent_result(
    *,
    job: QuestionJob,
    processor_provider: ProcessorProvider,
    ledger: RunLedger,
    publisher: SlackPublisher,
) -> AgentResponse | None:
    """Resume one graph run and emit only sanitized, monotonic progress events."""

    observation = await ledger.observe_run(job.agent_run_id)
    if observation.cancellation_requested:
        return None
    persisted_response = await ledger.get_persisted_agent_result(job.agent_run_id)
    if persisted_response is not None:
        return persisted_response

    final_response: AgentResponse | None = None
    async for event in processor_provider().run(
        question=job.question,
        conversation_id=job.conversation_id,
        agent_run_id=str(job.agent_run_id),
    ):
        observation = await ledger.observe_run(job.agent_run_id)
        if observation.cancellation_requested:
            return None
        if isinstance(event, ProgressEvent):
            # Progress updates are advisory UX only; final model result is authoritative.
            try:
                await publisher.publish_progress(job.agent_run_id, event)
            except Exception as exc:
                # A status update never makes grounded answer generation fail.
                logger.warning(
                    "slack_progress_publish_failed",
                    agent_run_id=str(job.agent_run_id),
                    progress_sequence=event.sequence,
                    exception_class=type(exc).__name__,
                )
            continue
        if isinstance(event, FinalAnswerEvent):
            final_response = event.response

    if final_response is None:
        raise RuntimeError("Question processor ended without a final answer")
    observation = await ledger.observe_run(job.agent_run_id)
    if observation.cancellation_requested:
        return None
    # Persist answer once; replayed steps must read this immutable artifact instead of recomputing.
    await ledger.persist_agent_result(job.agent_run_id, final_response)
    return final_response


def _original_question_job(event_data: dict[str, Any]) -> QuestionJob:
    original = event_data.get("event")
    if not isinstance(original, dict) or not isinstance(original.get("data"), dict):
        raise ValueError("Inngest system event has no original question payload")
    return QuestionJob.model_validate(original["data"])


def _original_slack_turn(event_data: dict[str, Any]) -> tuple[str, str]:
    """Return the original Slack event name and stable event ID from a router failure."""

    original = event_data.get("event")
    if not isinstance(original, dict) or not isinstance(original.get("data"), dict):
        raise ValueError("Inngest system event has no original Slack turn payload")
    event_name = str(original.get("name", ""))
    original_data = cast(dict[str, Any], original["data"])
    if event_name == QUESTION_RECEIVED_EVENT:
        event_id = QuestionJob.model_validate(original_data).event_id
    elif event_name == FOLLOW_UP_CANDIDATE_EVENT:
        event_id = FollowUpCandidateJob.model_validate(original_data).event_id
    else:
        raise ValueError(f"Unsupported original Slack turn event: {event_name}")
    return event_name, event_id


def _question_cancellation_job(ctx: inngest.Context) -> QuestionCancellationJob:
    if ctx.event.name == QUESTION_CANCELLED_EVENT:
        return QuestionCancellationJob.model_validate(ctx.event.data)
    original_job = _original_question_job(cast(dict[str, Any], ctx.event.data))
    return QuestionCancellationJob(
        event_id=f"inngest-cancel:{ctx.event.id}",
        team_id=original_job.team_id,
        channel_id=original_job.channel_id,
        user_id=original_job.user_id,
        thread_ts=original_job.thread_ts,
        event_ts=str(ctx.event.ts),
        streaming_message_ts=(),
        agent_run_id=original_job.agent_run_id,
        cancellation_accepted=False,
    )


async def _finalize_user_cancellation(
    ctx: inngest.Context,
    *,
    run_id: uuid.UUID,
    ledger: RunLedger,
    publisher: SlackPublisher,
) -> str:
    """Idempotently terminate Slack and the ledger when a persisted Stop wins."""

    delivery = await ledger.get_delivery(run_id)
    if delivery.delivery_status == DeliveryStatus.DELIVERED:
        persisted_response = await ledger.get_persisted_agent_result(run_id)
        if persisted_response is not None:
            await ctx.step.run(
                "recover-delivered-run",
                lambda: ledger.mark_succeeded(run_id, persisted_response),
            )
            return "succeeded"
    await ctx.step.run("publish-cancelled", lambda: publisher.publish_cancelled(run_id))
    await ctx.step.run("mark-run-cancelled", lambda: ledger.mark_cancelled(run_id))
    return "cancelled"


async def _recover_persisted_delivery(
    ctx: inngest.Context,
    *,
    step_prefix: str,
    run_id: uuid.UUID,
    response: AgentResponse | None,
    ledger: RunLedger,
    publisher: SlackPublisher,
) -> bool:
    """Resume the canonical persisted answer before considering a generic failure notice."""

    if response is None:
        return False
    delivery = await ledger.get_delivery(run_id)
    if delivery.delivery_status == DeliveryStatus.DELIVERED:
        await ctx.step.run(
            f"{step_prefix}-mark-run-completed",
            lambda: ledger.mark_succeeded(run_id, response),
        )
        return True
    if delivery.delivery_status not in {
        DeliveryStatus.PENDING,
        DeliveryStatus.DELIVERING,
    }:
        return False

    async def prepare_delivery() -> dict[str, Any]:
        prepared = await publisher.prepare_delivery(run_id, response)
        return prepared.model_dump(mode="json")

    prepared_payload = cast(
        dict[str, Any],
        await ctx.step.run(f"{step_prefix}-prepare-delivery", prepare_delivery),
    )
    prepared = PreparedDelivery.model_validate(prepared_payload)
    should_deliver = bool(
        await ctx.step.run(
            f"{step_prefix}-claim-delivery",
            lambda: publisher.begin_delivery(run_id),
        )
    )
    if not should_deliver:
        delivery = await ledger.get_delivery(run_id)
        if delivery.delivery_status != DeliveryStatus.DELIVERED:
            return False
        await ctx.step.run(
            f"{step_prefix}-mark-run-completed",
            lambda: ledger.mark_succeeded(run_id, response),
        )
        return True

    for part_number in range(1, len(prepared.parts) + 1):

        async def publish_part(
            payload: dict[str, Any],
            number: int,
        ) -> str:
            return await publisher.publish_delivery_part(
                run_id,
                PreparedDelivery.model_validate(payload),
                number,
            )

        try:
            await ctx.step.run(
                f"{step_prefix}-publish-answer-part-{part_number}",
                publish_part,
                prepared_payload,
                part_number,
            )
        except SlackDeliveryRejectedError as exc:
            logger.warning(
                "slack_canonical_delivery_rejected",
                agent_run_id=str(run_id),
                delivery_part_number=part_number,
                exception_class=type(exc).__name__,
            )
            return False

    await ctx.step.run(
        f"{step_prefix}-complete-delivery",
        lambda: publisher.complete_delivery(run_id),
    )
    await ctx.step.run(
        f"{step_prefix}-mark-run-completed",
        lambda: ledger.mark_succeeded(run_id, response),
    )
    return True


async def _has_partially_acknowledged_delivery(
    run_id: uuid.UUID,
    ledger: RunLedger,
) -> bool:
    manifest = await ledger.get_delivery_manifest(run_id)
    if manifest is None or not manifest.parts:
        return False
    acknowledged_count = sum(part.acknowledged_at is not None for part in manifest.parts)
    return 0 < acknowledged_count < len(manifest.parts)


def create_question_functions(
    client: inngest.Inngest,
    *,
    processor_provider: ProcessorProvider,
    responder_classifier: ResponderClassifier | None,
    ledger: RunLedger,
    publisher: SlackPublisher,
) -> list[Any]:
    """Register durable work at boundaries with one clear retry owner per effect."""

    model_concurrency = inngest.Concurrency(limit=8, key='"openai"', scope="env")
    progress_concurrency = inngest.Concurrency(
        limit=16,
        key='"slack_progress"',
        scope="env",
    )
    # This is a contention bound, not the ordering invariant. PostgreSQL's slack_turns
    # causal queue remains authoritative even while Inngest releases capacity at sleeps
    # and child-function invokes.
    conversation_concurrency = inngest.Concurrency(
        limit=1,
        key="event.data.conversation_id",
        scope="env",
    )
    # Inngest is still free to invoke multiple worker pipelines for different conversations.
    # This per-conversation lock preserves ordering where Slack replay or async wakeups could
    # otherwise re-enter stale turns out of causal sequence.

    @client.create_function(
        fn_id=QUESTION_FUNCTION_ID,
        name="Process Slack question",
        trigger=inngest.TriggerEvent(event=QUESTION_READY_EVENT),
        retries=3,
        idempotency="event.data.event_id",
        cancel=[
            inngest.Cancel(
                event=QUESTION_CANCELLED_EVENT,
                if_exp=(
                    "async.data.cancellation_accepted == true && "
                    "event.data.agent_run_id == async.data.agent_run_id"
                ),
                timeout=timedelta(hours=1),
            )
        ],
        timeouts=inngest.Timeouts(
            start=timedelta(minutes=5),
            finish=timedelta(minutes=15),
        ),
        concurrency=[model_concurrency],
    )
    async def process_slack_question(ctx: inngest.Context) -> dict[str, Any]:
        job = QuestionJob.model_validate(ctx.event.data)

        async def claim_run() -> dict[str, Any]:
            # Run claims serialize start/stop transitions: duplicate deliveries must resume or fail
            # closed instead of executing duplicate LLM calls.
            claim = await ledger.claim_run(job.agent_run_id)
            return {
                "status": claim.status.value,
                "should_process": claim.should_process,
                "cancellation_requested": claim.cancellation_requested,
            }

        claim_payload = cast(dict[str, Any], await ctx.step.run("claim-run", claim_run))
        if not bool(claim_payload["should_process"]):
            run_status = RunStatus(str(claim_payload["status"]))
            if bool(claim_payload["cancellation_requested"]) and run_status in {
                RunStatus.QUEUED,
                RunStatus.RUNNING,
            }:
                cancellation_status = await _finalize_user_cancellation(
                    ctx,
                    run_id=job.agent_run_id,
                    ledger=ledger,
                    publisher=publisher,
                )
                return {
                    "agent_run_id": str(job.agent_run_id),
                    "status": cancellation_status,
                }
            return {
                "agent_run_id": str(job.agent_run_id),
                "status": run_status.value,
            }

        try:
            # This step is best-effort. If stream setup cannot be completed yet, final answer
            # delivery can still continue later as a bounded fallback path.
            await _ensure_progress_surface(
                ctx,
                run_id=job.agent_run_id,
                publisher=publisher,
            )
        except Exception as exc:
            # Native progress is optional. Final delivery still reconciles the persisted
            # stream state and must not lose a grounded answer because the loader failed.
            logger.warning(
                "slack_progress_initialization_failed",
                agent_run_id=str(job.agent_run_id),
                conversation_id=job.conversation_id,
                exception_class=type(exc).__name__,
            )

        async def run_agent() -> dict[str, Any]:
            # The processor boundary encapsulates all model + retrieval work and is the main idempotent
            # work unit: replaying this step must reuse checkpoints, not branch a second run.
            response = await _stream_agent_result(
                job=job,
                processor_provider=processor_provider,
                ledger=ledger,
                publisher=publisher,
            )
            if response is None:
                return {"cancelled": True, "response": None}
            return {"cancelled": False, "response": response.model_dump(mode="json")}

        agent_payload = cast(dict[str, Any], await ctx.step.run("run-agent", run_agent))
        if bool(agent_payload["cancelled"]):
            cancellation_status = await _finalize_user_cancellation(
                ctx,
                run_id=job.agent_run_id,
                ledger=ledger,
                publisher=publisher,
            )
            return {
                "agent_run_id": str(job.agent_run_id),
                "status": cancellation_status,
            }
        response_payload = cast(dict[str, Any], agent_payload["response"])
        response = AgentResponse.model_validate(response_payload)

        async def check_before_delivery() -> bool:
            observation = await ledger.observe_run(job.agent_run_id)
            return observation.cancellation_requested

        cancellation_requested = bool(
            await ctx.step.run("check-cancellation-before-delivery", check_before_delivery)
        )
        if cancellation_requested:
            cancellation_status = await _finalize_user_cancellation(
                ctx,
                run_id=job.agent_run_id,
                ledger=ledger,
                publisher=publisher,
            )
            return {
                "agent_run_id": str(job.agent_run_id),
                "status": cancellation_status,
            }

        async def prepare_delivery() -> dict[str, Any]:
            prepared = await publisher.prepare_delivery(job.agent_run_id, response)
            return prepared.model_dump(mode="json")

        prepared_payload = cast(
            dict[str, Any],
            await ctx.step.run("prepare-delivery", prepare_delivery),
        )
        prepared = PreparedDelivery.model_validate(prepared_payload)
        should_deliver = bool(
            await ctx.step.run(
                "claim-delivery",
                lambda: publisher.begin_delivery(job.agent_run_id),
            )
        )
        if not should_deliver:
            delivery = await ledger.get_delivery(job.agent_run_id)
            if delivery.delivery_status == DeliveryStatus.DELIVERED:
                await ctx.step.run(
                    "recover-delivered-run",
                    lambda: ledger.mark_succeeded(job.agent_run_id, response),
                )
                return {
                    "agent_run_id": str(job.agent_run_id),
                    "status": "succeeded",
                }
            return {
                "agent_run_id": str(job.agent_run_id),
                "status": delivery.delivery_status.value,
            }

        for part_number in range(1, len(prepared.parts) + 1):

            async def publish_part(
                payload: dict[str, Any],
                number: int,
            ) -> str:
                return await publisher.publish_delivery_part(
                    job.agent_run_id,
                    PreparedDelivery.model_validate(payload),
                    number,
                )

            await ctx.step.run(
                f"publish-answer-part-{part_number}",
                publish_part,
                prepared_payload,
                part_number,
            )

        await ctx.step.run(
            "complete-delivery",
            lambda: publisher.complete_delivery(job.agent_run_id),
        )

        async def mark_completed(payload: dict[str, Any]) -> None:
            await ledger.mark_succeeded(
                job.agent_run_id,
                AgentResponse.model_validate(payload),
            )

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

    @client.create_function(
        fn_id=PROGRESS_FUNCTION_ID,
        name="Initialize Slack progress",
        trigger=inngest.TriggerEvent(event=QUESTION_READY_EVENT),
        retries=3,
        idempotency="event.data.event_id",
        timeouts=inngest.Timeouts(
            start=timedelta(minutes=1),
            finish=timedelta(minutes=2),
        ),
        concurrency=[progress_concurrency],
    )
    async def initialize_slack_progress(ctx: inngest.Context) -> dict[str, Any]:
        """Open the native loader independently of queued model capacity."""

        job = QuestionJob.model_validate(ctx.event.data)
        observation = await ledger.observe_run(job.agent_run_id)
        if observation.is_terminal or observation.cancellation_requested:
            return {
                "agent_run_id": str(job.agent_run_id),
                "status": observation.status.value,
            }
        timestamp = await _ensure_progress_surface(
            ctx,
            run_id=job.agent_run_id,
            publisher=publisher,
        )
        if timestamp is None:
            return {
                "agent_run_id": str(job.agent_run_id),
                "status": "deferred",
            }
        return {
            "agent_run_id": str(job.agent_run_id),
            "status": "initialized",
            "slack_message_ts": str(timestamp),
        }

    @client.create_function(
        fn_id=RESPONDER_CLASSIFIER_FUNCTION_ID,
        name="Classify Slack follow-up",
        trigger=inngest.TriggerEvent(event=RESPONDER_CLASSIFICATION_EVENT),
        retries=2,
        idempotency="event.data.event_id",
        timeouts=inngest.Timeouts(
            start=timedelta(minutes=5),
            finish=timedelta(minutes=2),
        ),
        concurrency=[model_concurrency],
    )
    async def classify_slack_follow_up(ctx: inngest.Context) -> dict[str, Any]:
        """Give semantic responder judgment its own globally bounded model slot."""

        payload = cast(dict[str, Any], ctx.event.data)
        request_payload = payload.get("request")
        if not isinstance(request_payload, dict):
            raise ValueError("Follow-up classification has no request payload")
        if responder_classifier is None:
            raise RuntimeError("Follow-up classifier is not configured")
        classification = await responder_classifier.classify(
            ResponderClassificationRequest.model_validate(request_payload)
        )
        return classification.model_dump(mode="json")

    @client.create_function(
        fn_id=TURN_ROUTER_FUNCTION_ID,
        name="Route Slack turn",
        trigger=[
            inngest.TriggerEvent(event=QUESTION_RECEIVED_EVENT),
            inngest.TriggerEvent(event=FOLLOW_UP_CANDIDATE_EVENT),
        ],
        retries=3,
        timeouts=inngest.Timeouts(
            start=timedelta(minutes=5),
            finish=timedelta(minutes=75),
        ),
        concurrency=[conversation_concurrency],
    )
    async def route_slack_turn(ctx: inngest.Context) -> dict[str, Any]:
        """Serialize one persisted Slack turn through routing, execution, and delivery."""

        candidate: FollowUpCandidateJob | None = None
        # The same durable router owns both explicit mentions and follow-up candidates so they
        # share identical ordering and completion semantics.
        if ctx.event.name == QUESTION_RECEIVED_EVENT:
            question = QuestionJob.model_validate(ctx.event.data)
            turn_kind = SlackTurnKind.EXPLICIT_MENTION
            event_id = question.event_id
            team_id = question.team_id
            channel_id = question.channel_id
            user_id = question.user_id
            message_ts = question.message_ts
            thread_ts = question.thread_ts
            conversation_id = question.conversation_id
        elif ctx.event.name == FOLLOW_UP_CANDIDATE_EVENT:
            candidate = FollowUpCandidateJob.model_validate(ctx.event.data)
            question = None
            turn_kind = SlackTurnKind.FOLLOW_UP
            event_id = candidate.event_id
            team_id = candidate.team_id
            channel_id = candidate.channel_id
            user_id = candidate.user_id
            message_ts = candidate.message_ts
            thread_ts = candidate.thread_ts
            conversation_id = candidate.conversation_id
        else:
            raise ValueError(f"Unsupported Slack turn event: {ctx.event.name}")

        async def ensure_turn() -> dict[str, Any]:
            # ensure_turn creates/reads queue state without taking processing ownership.
            result = await ledger.ensure_turn(
                event_id=event_id,
                team_id=team_id,
                channel_id=channel_id,
                user_id=user_id,
                message_ts=message_ts,
                thread_ts=thread_ts,
                kind=turn_kind,
            )
            return {
                "status": result.turn.status.value,
                "was_created": result.was_created,
            }

        await ctx.step.run("ensure-turn", ensure_turn)
        # ensure_turn is durable and idempotent; ownership is intentionally acquired in
        # claim-turn so retries cannot create duplicate run work.

        turn_status: SlackTurnStatus | None = None
        linked_run_id: uuid.UUID | None = None
        for attempt in range(1, TURN_WAIT_ATTEMPTS + 1):
            # Wait-and-retry gives Inngest time to advance older queue turns before this turn claims ownership.

            async def claim_turn() -> dict[str, Any]:
                # Only the causal head of the conversation queue keeps should_process=True.
                claim = await ledger.claim_turn(event_id)
                return {
                    "agent_run_id": (
                        str(claim.turn.agent_run_id)
                        if claim.turn.agent_run_id is not None
                        else None
                    ),
                    "should_process": claim.should_process,
                    "status": claim.turn.status.value,
                    "was_claimed": claim.was_claimed,
                }

            claim_payload = cast(
                dict[str, Any],
                await ctx.step.run(f"claim-turn-{attempt}", claim_turn),
            )
            turn_status = SlackTurnStatus(str(claim_payload["status"]))
            run_id_value = claim_payload.get("agent_run_id")
            linked_run_id = uuid.UUID(str(run_id_value)) if run_id_value else None
            if bool(claim_payload["should_process"]):
                break
            if turn_status in {
                SlackTurnStatus.ROUTED,
                SlackTurnStatus.SUPPRESSED,
                SlackTurnStatus.FAILED,
            }:
                return {
                    "agent_run_id": str(linked_run_id) if linked_run_id is not None else None,
                    "event_id": event_id,
                    "status": turn_status.value,
                }
            await ctx.step.sleep(f"wait-for-causal-turn-{attempt}", TURN_WAIT_INTERVAL)
        else:
            raise RuntimeError("Timed out waiting for the causal Slack turn head")

        async def complete_turn(target: SlackTurnStatus) -> None:
            await ledger.complete_turn(event_id, target)

        if candidate is not None:
            # Follow-up turns need both an existing conversation owner and an active
            # classifier; otherwise they must be suppressed to avoid interruptive false positives.

            async def load_latest_agent_response() -> dict[str, Any] | None:
                response = await ledger.get_latest_delivered_agent_response(
                    candidate.team_id,
                    candidate.channel_id,
                    candidate.thread_ts,
                )
                return response.model_dump(mode="json") if response is not None else None

            latest_response_payload = cast(
                dict[str, Any] | None,
                await ctx.step.run(
                    "load-latest-delivered-agent-response",
                    load_latest_agent_response,
                ),
            )
            if latest_response_payload is None or responder_classifier is None:
                status = (
                    "not_agent_owned" if latest_response_payload is None else "classifier_disabled"
                )
                await ctx.step.run(
                    "complete-suppressed-turn",
                    complete_turn,
                    SlackTurnStatus.SUPPRESSED,
                )
                return {"event_id": event_id, "status": status}

            latest_response = AgentResponse.model_validate(latest_response_payload)
            clarification_answer = latest_response.answer.strip()
            clarification_question = (
                clarification_answer[:8_000]
                if latest_response.requires_user_input and clarification_answer
                else None
            )
            classification_request = ResponderClassificationRequest(
                thread=SlackThreadIdentity(
                    team_id=candidate.team_id,
                    channel_id=candidate.channel_id,
                    thread_ts=candidate.thread_ts,
                ),
                user_id=candidate.user_id,
                message_text=candidate.message_text,
                last_agent_clarification_question=clarification_question,
            )
            classification_payload = cast(
                dict[str, Any],
                await ctx.step.invoke(
                    "classify-responder",
                    function=classify_slack_follow_up,
                    data={
                        "event_id": candidate.event_id,
                        "request": classification_request.model_dump(mode="json"),
                    },
                    timeout=timedelta(minutes=2),
                ),
            )
            decision = decide_responder_classification(
                ResponderClassification.model_validate(classification_payload)
            )
            if not decision.should_respond:
                # Suppressed follow-ups must still be terminally marked so queued later turns
                # can continue through ordering as expected.
                await ctx.step.run(
                    "complete-suppressed-turn",
                    complete_turn,
                    SlackTurnStatus.SUPPRESSED,
                )
                logger.info(
                    "slack_follow_up_suppressed",
                    candidate_id=str(candidate.candidate_id),
                    conversation_id=candidate.conversation_id,
                    routing_reason=decision.reason.value,
                )
                return {"event_id": event_id, "status": decision.reason.value}
            question = QuestionJob(
                agent_run_id=candidate.candidate_id,
                event_id=candidate.event_id,
                team_id=candidate.team_id,
                channel_id=candidate.channel_id,
                user_id=candidate.user_id,
                message_ts=candidate.message_ts,
                thread_ts=candidate.thread_ts,
                question=candidate.message_text,
            )

        if question is None:
            raise RuntimeError("Accepted Slack turn has no question")

        async def create_linked_run() -> dict[str, Any]:
            run_id, is_new_run = await ledger.create_queued_for_turn(question, event_id)
            linked_question = question.model_copy(update={"agent_run_id": run_id})
            return {
                "is_new_run": is_new_run,
                "question": linked_question.model_dump(mode="json"),
            }

        linked_payload = cast(
            dict[str, Any],
            await ctx.step.run("create-linked-run", create_linked_run),
        )
        linked_question = QuestionJob.model_validate(linked_payload["question"])
        linked_run_id = linked_question.agent_run_id
        question_payload = linked_question.model_dump(mode="json")

        try:
            # Progress initialization is separated to avoid stalling question execution on
            # transient progress-surface failures.
            await ctx.step.invoke(
                "initialize-progress",
                function=initialize_slack_progress,
                data=question_payload,
                timeout=timedelta(minutes=3),
            )
        except Exception as exc:
            logger.warning(
                "slack_progress_initializer_exhausted",
                agent_run_id=str(linked_run_id),
                conversation_id=conversation_id,
                exception_class=type(exc).__name__,
            )
        await ctx.step.invoke(
            "process-question",
            function=process_slack_question,
            data=question_payload,
            timeout=timedelta(minutes=20),
        )

        async def verify_terminal_run() -> str:
            observation = await ledger.observe_run(linked_run_id)
            if not observation.is_terminal:
                raise RuntimeError("Invoked Slack question returned before its run became terminal")
            return observation.status.value

        run_status = str(await ctx.step.run("verify-terminal-run", verify_terminal_run))
        await ctx.step.run(
            "complete-routed-turn",
            complete_turn,
            SlackTurnStatus.ROUTED,
        )
        logger.info(
            "slack_turn_routed",
            agent_run_id=str(linked_run_id),
            conversation_id=conversation_id,
            slack_event_id=event_id,
            run_status=run_status,
            turn_kind=turn_kind.value,
        )
        return {
            "agent_run_id": str(linked_run_id),
            "event_id": event_id,
            "status": run_status,
        }

    @client.create_function(
        fn_id=TURN_FAILURE_CLEANUP_FUNCTION_ID,
        name="Clean up failed Slack turn",
        trigger=[
            inngest.TriggerEvent(
                event=FUNCTION_FAILED_EVENT,
                expression=(
                    f"event.data.function_id == '{TURN_ROUTER_FULLY_QUALIFIED_FUNCTION_ID}'"
                ),
            ),
            inngest.TriggerEvent(
                event=FUNCTION_CANCELLED_EVENT,
                expression=(
                    f"event.data.function_id == '{TURN_ROUTER_FULLY_QUALIFIED_FUNCTION_ID}'"
                ),
            ),
        ],
        retries=3,
        timeouts=inngest.Timeouts(finish=timedelta(minutes=12)),
    )
    async def cleanup_failed_turn(ctx: inngest.Context) -> dict[str, Any]:
        """Release the causal queue only after any linked run reaches a terminal state."""

        _event_name, event_id = _original_slack_turn(cast(dict[str, Any], ctx.event.data))

        async def load_turn() -> dict[str, Any] | None:
            turn = await ledger.get_turn(event_id)
            if turn is None:
                return None
            return {
                "agent_run_id": (str(turn.agent_run_id) if turn.agent_run_id is not None else None),
                "status": turn.status.value,
            }

        turn_payload = cast(
            dict[str, Any] | None,
            await ctx.step.run("load-failed-turn", load_turn),
        )
        if turn_payload is None:
            raise RuntimeError("Failed Slack turn was not durably persisted")
        turn_status = SlackTurnStatus(str(turn_payload["status"]))
        if turn_status in {
            SlackTurnStatus.ROUTED,
            SlackTurnStatus.SUPPRESSED,
            SlackTurnStatus.FAILED,
        }:
            return {"event_id": event_id, "status": turn_status.value}

        if turn_status is SlackTurnStatus.PENDING:
            # A pending state means there is a race to the front of the same thread queue.
            # Wait rather than force-closing so causality remains monotonic.
            for attempt in range(1, 21):

                async def claim_failed_turn() -> dict[str, Any]:
                    claim = await ledger.claim_turn(event_id)
                    return {
                        "agent_run_id": (
                            str(claim.turn.agent_run_id)
                            if claim.turn.agent_run_id is not None
                            else None
                        ),
                        "should_process": claim.should_process,
                        "status": claim.turn.status.value,
                    }

                turn_payload = cast(
                    dict[str, Any],
                    await ctx.step.run(
                        f"claim-failed-turn-{attempt}",
                        claim_failed_turn,
                    ),
                )
                turn_status = SlackTurnStatus(str(turn_payload["status"]))
                if bool(turn_payload["should_process"]):
                    break
                if turn_status in {
                    SlackTurnStatus.ROUTED,
                    SlackTurnStatus.SUPPRESSED,
                    SlackTurnStatus.FAILED,
                }:
                    return {"event_id": event_id, "status": turn_status.value}
                await ctx.step.sleep(
                    f"wait-to-clean-failed-turn-{attempt}",
                    TURN_WAIT_INTERVAL,
                )
            else:
                raise RuntimeError("Failed Slack turn is still blocked by an earlier turn")

        linked_run_value = turn_payload.get("agent_run_id")
        if linked_run_value is None:
            await ctx.step.run(
                "complete-unlinked-failed-turn",
                lambda: ledger.complete_turn(event_id, SlackTurnStatus.FAILED),
            )
            return {"event_id": event_id, "status": SlackTurnStatus.FAILED.value}

        linked_run_id = uuid.UUID(str(linked_run_value))
        for attempt in range(1, 21):
            # The failed turn can only be closed as routed when its run reaches a terminal status.

            async def observe_linked_run() -> dict[str, Any]:
                observation = await ledger.observe_run(linked_run_id)
                return {
                    "cancellation_requested": observation.cancellation_requested,
                    "status": observation.status.value,
                }

            observation_payload = cast(
                dict[str, Any],
                await ctx.step.run(
                    f"observe-failed-turn-run-{attempt}",
                    observe_linked_run,
                ),
            )
            run_status = RunStatus(str(observation_payload["status"]))
            if run_status in {
                RunStatus.SUCCEEDED,
                RunStatus.FAILED,
                RunStatus.CANCELLED,
            }:
                await ctx.step.run(
                    "complete-failed-routed-turn",
                    lambda: ledger.complete_turn(event_id, SlackTurnStatus.ROUTED),
                )
                return {
                    "agent_run_id": str(linked_run_id),
                    "event_id": event_id,
                    "status": run_status.value,
                }
            await ctx.step.sleep(
                f"wait-for-linked-run-cleanup-{attempt}",
                TURN_WAIT_INTERVAL,
            )
        raise RuntimeError("Linked run did not become terminal during turn cleanup")

    @client.create_function(
        fn_id=FAILURE_CLEANUP_FUNCTION_ID,
        name="Clean up failed Slack question",
        trigger=inngest.TriggerEvent(
            event=FUNCTION_FAILED_EVENT,
            expression=(f"event.data.function_id == '{QUESTION_FULLY_QUALIFIED_FUNCTION_ID}'"),
        ),
        retries=3,
        timeouts=inngest.Timeouts(finish=timedelta(minutes=5)),
    )
    async def cleanup_failed_question(ctx: inngest.Context) -> dict[str, Any]:
        job = _original_question_job(cast(dict[str, Any], ctx.event.data))
        observation = await ledger.observe_run(job.agent_run_id)
        if observation.is_terminal or observation.cancellation_requested:
            return {"agent_run_id": str(job.agent_run_id), "status": observation.status.value}

        persisted_response = await ledger.get_persisted_agent_result(job.agent_run_id)
        if await _recover_persisted_delivery(
            ctx,
            step_prefix="failure-recovery",
            run_id=job.agent_run_id,
            response=persisted_response,
            ledger=ledger,
            publisher=publisher,
        ):
            return {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}

        if await _has_partially_acknowledged_delivery(job.agent_run_id, ledger):
            await ctx.step.run(
                "publish-incomplete-delivery-notice",
                lambda: publisher.publish_incomplete_delivery_notice(job.agent_run_id),
            )
            await ctx.step.run(
                "mark-incomplete-delivery-failed",
                lambda: ledger.mark_failed(
                    job.agent_run_id,
                    code="slack_delivery_incomplete",
                    message="Slack delivery ended after only some answer parts were acknowledged.",
                ),
            )
            return {"agent_run_id": str(job.agent_run_id), "status": "failed"}

        await ctx.step.run(
            "publish-safe-error",
            lambda: publisher.publish_safe_error(job.agent_run_id),
        )
        await ctx.step.run(
            "mark-run-failed",
            lambda: ledger.mark_failed(
                job.agent_run_id,
                code="inngest_retries_exhausted",
                message="Background processing failed after retry exhaustion.",
            ),
        )
        return {"agent_run_id": str(job.agent_run_id), "status": "failed"}

    @client.create_function(
        fn_id=CANCELLATION_CLEANUP_FUNCTION_ID,
        name="Clean up cancelled Slack question",
        trigger=[
            inngest.TriggerEvent(event=QUESTION_CANCELLED_EVENT),
            inngest.TriggerEvent(
                event=FUNCTION_CANCELLED_EVENT,
                expression=(f"event.data.function_id == '{QUESTION_FULLY_QUALIFIED_FUNCTION_ID}'"),
            ),
        ],
        retries=3,
        timeouts=inngest.Timeouts(finish=timedelta(minutes=5)),
    )
    async def cleanup_cancelled_question(ctx: inngest.Context) -> dict[str, Any]:
        cancellation = _question_cancellation_job(ctx)
        run_id = cancellation.agent_run_id
        if run_id is None:
            # A delayed or duplicate Stop event is not allowed to rewrite the visible
            # status of a newer turn (or a completed clarification) when it cannot be
            # tied to a ledger run. The run that owns the session owns its final status.
            return {"agent_run_id": None, "status": "no_active_run"}

        observation = await ledger.observe_run(run_id)
        if observation.status == RunStatus.SUCCEEDED:
            # Successful finalization already chose active versus suspended. Replaying
            # an older Stop event must not overwrite that disposition-owned status.
            return {"agent_run_id": str(run_id), "status": "succeeded"}
        if observation.status in {RunStatus.FAILED, RunStatus.CANCELLED}:
            return {"agent_run_id": str(run_id), "status": observation.status.value}

        if ctx.event.name == QUESTION_CANCELLED_EVENT and not cancellation.cancellation_accepted:
            # Delivery already won (or the event matched no cancellable work). Do not falsely
            # claim the user stopped an answer that is now being finalized.
            return {"agent_run_id": str(run_id), "status": "cancellation_rejected"}

        if ctx.event.name == FUNCTION_CANCELLED_EVENT and not observation.cancellation_requested:
            persisted_response = (
                await ledger.get_persisted_agent_result(run_id)
                if observation.status == RunStatus.RUNNING
                else None
            )
            if await _recover_persisted_delivery(
                ctx,
                step_prefix="system-cancel-recovery",
                run_id=run_id,
                response=persisted_response,
                ledger=ledger,
                publisher=publisher,
            ):
                return {"agent_run_id": str(run_id), "status": "succeeded"}
            if await _has_partially_acknowledged_delivery(run_id, ledger):
                await ctx.step.run(
                    "publish-system-cancel-incomplete-notice",
                    lambda: publisher.publish_incomplete_delivery_notice(run_id),
                )
                await ctx.step.run(
                    "mark-system-cancel-incomplete-failed",
                    lambda: ledger.mark_failed(
                        run_id,
                        code="slack_delivery_incomplete",
                        message=(
                            "Background processing was cancelled after partial Slack delivery."
                        ),
                    ),
                )
                return {"agent_run_id": str(run_id), "status": "failed"}
            await ctx.step.run(
                "publish-system-cancellation-error",
                lambda: publisher.publish_safe_error(run_id),
            )
            await ctx.step.run(
                "mark-system-cancellation-failed",
                lambda: ledger.mark_failed(
                    run_id,
                    code="inngest_function_cancelled",
                    message="Background processing was cancelled before completion.",
                ),
            )
            return {"agent_run_id": str(run_id), "status": "failed"}

        async def request_cancellation() -> dict[str, Any]:
            requested = await ledger.request_cancellation(run_id)
            return {
                "status": requested.status.value,
                "cancellation_requested": requested.cancellation_requested,
            }

        await ctx.step.run("request-cancellation", request_cancellation)
        delivery = await ledger.get_delivery(run_id)
        persisted_response = (
            await ledger.get_persisted_agent_result(run_id)
            if observation.status == RunStatus.RUNNING
            else None
        )
        if delivery.delivery_status == DeliveryStatus.DELIVERED and persisted_response is not None:
            await ctx.step.run(
                "recover-delivered-run",
                lambda: ledger.mark_succeeded(run_id, persisted_response),
            )
            return {"agent_run_id": str(run_id), "status": "succeeded"}

        status = await _finalize_user_cancellation(
            ctx,
            run_id=run_id,
            ledger=ledger,
            publisher=publisher,
        )
        return {"agent_run_id": str(run_id), "status": status}

    @client.create_function(
        fn_id="finalize-unreconciled-slack-run",
        name="Finalize an unreconciled Slack run",
        trigger=[
            inngest.TriggerEvent(
                event=system_event,
                expression=(
                    "event.data.function_id == "
                    f"'{FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID}' || "
                    "event.data.function_id == "
                    f"'{CANCELLATION_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID}' || "
                    "event.data.function_id == "
                    f"'{TURN_FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID}'"
                ),
            )
            for system_event in (FUNCTION_FAILED_EVENT, FUNCTION_CANCELLED_EVENT)
        ],
        retries=3,
        timeouts=inngest.Timeouts(finish=timedelta(minutes=5)),
    )
    async def finalize_unreconciled_run(ctx: inngest.Context) -> dict[str, Any]:
        """Release a run after bounded Slack reconciliation can no longer prove the write."""

        failed_event = cast(dict[str, Any], ctx.event.data)
        failed_function_id = str(failed_event.get("function_id", ""))
        original_event = failed_event.get("event")
        if not isinstance(original_event, dict) or not isinstance(original_event.get("data"), dict):
            raise ValueError("Inngest cleanup failure has no original event payload")
        original_data = cast(dict[str, Any], original_event["data"])

        if failed_function_id == TURN_FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID:
            _event_name, event_id = _original_slack_turn(original_data)
            turn = await ledger.get_turn(event_id)
            if turn is None:
                raise RuntimeError("Unreconciled Slack turn was not durably persisted")
            if turn.status in {
                SlackTurnStatus.ROUTED,
                SlackTurnStatus.SUPPRESSED,
                SlackTurnStatus.FAILED,
            }:
                return {"event_id": event_id, "status": turn.status.value}
            if turn.status is SlackTurnStatus.PENDING:
                claim = await ledger.claim_turn(event_id)
                if not claim.should_process:
                    raise RuntimeError(
                        "Unreconciled Slack turn is still blocked by an earlier turn"
                    )
                turn = claim.turn
            run_id = turn.agent_run_id
            if run_id is None:
                await ctx.step.run(
                    "fail-unlinked-unreconciled-turn",
                    lambda: ledger.complete_turn(event_id, SlackTurnStatus.FAILED),
                )
                return {"event_id": event_id, "status": SlackTurnStatus.FAILED.value}

            linked_turn_run_id = run_id
            observation = await ledger.observe_run(linked_turn_run_id)
            if not observation.is_terminal:
                if observation.cancellation_requested:
                    await ctx.step.run(
                        "abandon-unreconciled-turn-cancellation",
                        lambda: publisher.abandon_unreconciled_cancellation(linked_turn_run_id),
                    )
                    await ctx.step.run(
                        "cancel-unreconciled-turn-run",
                        lambda: ledger.mark_cancelled(linked_turn_run_id),
                    )
                    run_status = RunStatus.CANCELLED
                else:
                    await ctx.step.run(
                        "abandon-unreconciled-turn-failure",
                        lambda: publisher.abandon_unreconciled_failure(linked_turn_run_id),
                    )
                    await ctx.step.run(
                        "fail-unreconciled-turn-run",
                        lambda: ledger.mark_failed(
                            linked_turn_run_id,
                            code="turn_reconciliation_exhausted",
                            message=("The Slack turn and its linked run could not be reconciled."),
                        ),
                    )
                    run_status = RunStatus.FAILED
            else:
                run_status = observation.status
            await ctx.step.run(
                "complete-unreconciled-routed-turn",
                lambda: ledger.complete_turn(event_id, SlackTurnStatus.ROUTED),
            )
            logger.error(
                "slack_turn_reconciliation_exhausted",
                agent_run_id=str(linked_turn_run_id),
                slack_event_id=event_id,
                error_code="turn_reconciliation_exhausted",
            )
            return {
                "agent_run_id": str(linked_turn_run_id),
                "event_id": event_id,
                "status": run_status.value,
            }

        if failed_function_id == CANCELLATION_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID:
            try:
                cancellation = QuestionCancellationJob.model_validate(original_data)
            except ValueError:
                job = _original_question_job(original_data)
                observation = await ledger.observe_run(job.agent_run_id)
                if observation.is_terminal:
                    return {
                        "agent_run_id": str(job.agent_run_id),
                        "status": observation.status.value,
                    }
                if observation.cancellation_requested:
                    await ctx.step.run(
                        "abandon-unreconciled-cancellation",
                        lambda: publisher.abandon_unreconciled_cancellation(job.agent_run_id),
                    )
                    await ctx.step.run(
                        "mark-unreconciled-run-cancelled",
                        lambda: ledger.mark_cancelled(job.agent_run_id),
                    )
                    return {
                        "agent_run_id": str(job.agent_run_id),
                        "status": "cancelled_unconfirmed",
                    }
                await ctx.step.run(
                    "abandon-unreconciled-system-cancellation",
                    lambda: publisher.abandon_unreconciled_failure(job.agent_run_id),
                )
                await ctx.step.run(
                    "mark-unreconciled-system-cancellation-failed",
                    lambda: ledger.mark_failed(
                        job.agent_run_id,
                        code="system_cancellation_reconciliation_exhausted",
                        message=("System cancellation could not be reconciled after all attempts."),
                    ),
                )
                logger.error(
                    "slack_system_cancellation_reconciliation_exhausted",
                    agent_run_id=str(job.agent_run_id),
                    error_code="system_cancellation_reconciliation_exhausted",
                )
                return {
                    "agent_run_id": str(job.agent_run_id),
                    "status": "failed_unconfirmed",
                }
            run_id = cancellation.agent_run_id
            if run_id is None or not cancellation.cancellation_accepted:
                return {"agent_run_id": None, "status": "cancellation_not_accepted"}
            observation = await ledger.observe_run(run_id)
            if observation.is_terminal:
                return {"agent_run_id": str(run_id), "status": observation.status.value}
            if not observation.cancellation_requested:
                logger.error(
                    "slack_cancellation_reconciliation_inconsistent",
                    agent_run_id=str(run_id),
                    error_code="cancellation_intent_missing",
                )
                return {"agent_run_id": str(run_id), "status": "cancellation_intent_missing"}
            await ctx.step.run(
                "abandon-unreconciled-cancellation",
                lambda: publisher.abandon_unreconciled_cancellation(run_id),
            )
            await ctx.step.run(
                "mark-unreconciled-run-cancelled",
                lambda: ledger.mark_cancelled(run_id),
            )
            logger.error(
                "slack_cancellation_reconciliation_exhausted",
                agent_run_id=str(run_id),
                error_code="slack_cancellation_reconciliation_exhausted",
            )
            return {"agent_run_id": str(run_id), "status": "cancelled_unconfirmed"}

        if failed_function_id != FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID:
            return {"agent_run_id": None, "status": "unmatched_cleanup"}
        job = _original_question_job(original_data)
        observation = await ledger.observe_run(job.agent_run_id)
        if observation.is_terminal:
            return {"agent_run_id": str(job.agent_run_id), "status": observation.status.value}
        persisted_response = await ledger.get_persisted_agent_result(job.agent_run_id)
        delivery = await ledger.get_delivery(job.agent_run_id)
        if delivery.delivery_status == DeliveryStatus.DELIVERED and persisted_response is not None:
            await ctx.step.run(
                "mark-late-reconciled-run-completed",
                lambda: ledger.mark_succeeded(job.agent_run_id, persisted_response),
            )
            return {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
        await ctx.step.run(
            "abandon-unreconciled-failure",
            lambda: publisher.abandon_unreconciled_failure(job.agent_run_id),
        )
        await ctx.step.run(
            "mark-unreconciled-run-failed",
            lambda: ledger.mark_failed(
                job.agent_run_id,
                code="slack_delivery_reconciliation_exhausted",
                message=("Slack delivery could not be proven after all reconciliation attempts."),
            ),
        )
        logger.error(
            "slack_delivery_reconciliation_exhausted",
            agent_run_id=str(job.agent_run_id),
            error_code="slack_delivery_reconciliation_exhausted",
        )
        return {"agent_run_id": str(job.agent_run_id), "status": "failed_unconfirmed"}

    return [
        process_slack_question,
        initialize_slack_progress,
        classify_slack_follow_up,
        route_slack_turn,
        cleanup_failed_turn,
        cleanup_failed_question,
        cleanup_cancelled_question,
        finalize_unreconciled_run,
    ]
