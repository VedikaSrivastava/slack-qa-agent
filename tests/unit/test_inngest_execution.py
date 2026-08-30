from __future__ import annotations

import hashlib
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from types import SimpleNamespace
from typing import Any, cast

import pytest

from knowledge_assistant.agent.models import (
    AgentResponse,
    FinalAnswerEvent,
    ProgressEvent,
    ProgressStage,
    QuestionDisposition,
)
from knowledge_assistant.application.question_processor import StreamingQuestionProcessor
from knowledge_assistant.execution.inngest import (
    CANCELLATION_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
    FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
    FUNCTION_CANCELLED_EVENT,
    FUNCTION_FAILED_EVENT,
    TURN_FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
    create_question_functions,
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
    ResponderDecision,
)
from knowledge_assistant.persistence.models import (
    DeliveryStatus,
    RunStatus,
    SlackTurnKind,
    SlackTurnStatus,
)
from knowledge_assistant.persistence.repositories import (
    DeliveryManifest,
    DeliveryPartState,
    DeliveryState,
    RunClaim,
    RunLedger,
    RunObservation,
    SlackTurnClaim,
    SlackTurnEnsureResult,
    SlackTurnRecord,
)


class FakeProcessor:
    def __init__(self, response: AgentResponse, events: list[str]) -> None:
        self.response = response
        self.events = events
        self.call_count = 0
        self.questions: list[str] = []

    async def answer(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AgentResponse:
        del question, conversation_id, agent_run_id
        self.call_count += 1
        self.events.append("answer")
        return self.response


class FakeStreamingProcessor(FakeProcessor):
    def run(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AsyncIterator[ProgressEvent | FinalAnswerEvent]:
        del conversation_id
        self.questions.append(question)

        async def events() -> AsyncIterator[ProgressEvent | FinalAnswerEvent]:
            self.call_count += 1
            self.events.append("graph")
            yield ProgressEvent(
                agent_run_id=agent_run_id,
                event_id=f"{agent_run_id}:1",
                sequence=1,
                stage=ProgressStage.SEARCHING,
            )
            yield FinalAnswerEvent(
                agent_run_id=agent_run_id,
                event_id=f"{agent_run_id}:final",
                response=self.response,
            )

        return events()


class FakeResultLedger:
    def __init__(
        self,
        persisted_response: AgentResponse | None,
        events: list[str],
    ) -> None:
        self.persisted_response = persisted_response
        self.events = events

    async def get_persisted_agent_result(self, _run_id: object) -> AgentResponse | None:
        self.events.append("load")
        return self.persisted_response

    async def persist_agent_result(self, _run_id: object, response: AgentResponse) -> None:
        self.events.append("persist")
        self.persisted_response = response


class FakeWorkflowLedger(FakeResultLedger):
    def __init__(self, events: list[str]) -> None:
        super().__init__(None, events)
        self.status = RunStatus.QUEUED
        self.cancellation_requested = False
        self.delivery_completed = False
        self.delivery_status = DeliveryStatus.PENDING
        self.manifest: DeliveryManifest | None = None
        self.latest_delivered_response: AgentResponse | None = None
        self.queued_jobs: list[QuestionJob] = []
        self.turns: dict[str, SlackTurnRecord] = {}
        self.turn_claim_scripts: dict[str, list[bool]] = {}
        self.run_observation_scripts: dict[object, list[RunObservation]] = {}
        self.cancelled_run_ids: set[object] = set()

    def script_turn_claims(self, event_id: str, *outcomes: bool) -> None:
        self.turn_claim_scripts[event_id] = list(outcomes)

    def script_run_observations(
        self,
        run_id: object,
        *observations: RunObservation,
    ) -> None:
        self.run_observation_scripts[run_id] = list(observations)

    async def ensure_turn(
        self,
        *,
        event_id: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        message_ts: str,
        thread_ts: str,
        message_text: str = "",
        kind: SlackTurnKind,
    ) -> SlackTurnEnsureResult:
        existing = self.turns.get(event_id)
        if existing is not None:
            return SlackTurnEnsureResult(existing, was_created=False)
        now = datetime.now(UTC)
        turn = SlackTurnRecord(
            event_id=event_id,
            team_id=team_id,
            channel_id=channel_id,
            user_id=user_id,
            message_ts=message_ts,
            thread_ts=thread_ts,
            message_text=message_text,
            conversation_id=f"{team_id}:{channel_id}:{thread_ts}",
            kind=kind,
            status=SlackTurnStatus.PENDING,
            agent_run_id=None,
            created_at=now,
            claimed_at=None,
            completed_at=None,
        )
        self.turns[event_id] = turn
        self.events.append(f"ensure_turn:{event_id}:{kind.value}")
        return SlackTurnEnsureResult(turn, was_created=True)

    async def get_recent_thread_recovery_messages(
        self,
        *,
        conversation_id: str,
        before_message_ts: str,
        limit: int,
    ) -> list[str]:
        recovered: list[tuple[Decimal, str]] = []
        for turn in self.turns.values():
            if turn.conversation_id != conversation_id:
                continue
            if Decimal(turn.message_ts) >= Decimal(before_message_ts):
                continue
            if turn.kind is SlackTurnKind.FOLLOW_UP and turn.status is SlackTurnStatus.SUPPRESSED:
                recovered.append((Decimal(turn.message_ts), turn.message_text))
                continue
            if (
                turn.kind is SlackTurnKind.EXPLICIT_MENTION
                and turn.agent_run_id is not None
                and turn.agent_run_id in self.cancelled_run_ids
            ):
                recovered.append((Decimal(turn.message_ts), turn.message_text))
        recovered.sort(key=lambda item: item[0])
        return [message for _, message in recovered[-limit:]]

    async def claim_turn(self, event_id: str) -> SlackTurnClaim:
        turn = self.turns[event_id]
        scripted_outcomes = self.turn_claim_scripts.get(event_id)
        if scripted_outcomes:
            should_process = scripted_outcomes.pop(0)
        else:
            processing_turn = next(
                (
                    candidate
                    for candidate in self.turns.values()
                    if candidate.conversation_id == turn.conversation_id
                    and candidate.status is SlackTurnStatus.PROCESSING
                ),
                None,
            )
            if processing_turn is not None:
                should_process = processing_turn.event_id == event_id
            else:
                pending_turns = [
                    candidate
                    for candidate in self.turns.values()
                    if candidate.conversation_id == turn.conversation_id
                    and candidate.status is SlackTurnStatus.PENDING
                ]
                head = min(
                    pending_turns,
                    key=lambda candidate: (
                        Decimal(candidate.message_ts),
                        candidate.created_at,
                        candidate.event_id,
                    ),
                    default=None,
                )
                should_process = head is not None and head.event_id == event_id

        was_claimed = should_process and turn.status is SlackTurnStatus.PENDING
        if was_claimed:
            turn = replace(
                turn,
                status=SlackTurnStatus.PROCESSING,
                claimed_at=datetime.now(UTC),
            )
            self.turns[event_id] = turn
        outcome = "claimed" if should_process else "blocked"
        self.events.append(f"claim_turn:{event_id}:{outcome}")
        return SlackTurnClaim(
            turn=turn,
            should_process=should_process,
            was_claimed=was_claimed,
        )

    async def create_queued_for_turn(
        self,
        job: QuestionJob,
        turn_event_id: str,
    ) -> tuple[object, bool]:
        turn = self.turns[turn_event_id]
        run_id = turn.agent_run_id or job.agent_run_id
        is_new_run = turn.agent_run_id is None
        self.turns[turn_event_id] = replace(turn, agent_run_id=run_id)
        linked_job = job.model_copy(update={"agent_run_id": run_id})
        if is_new_run:
            self.status = RunStatus.QUEUED
            self.cancellation_requested = False
            self.delivery_completed = False
            self.delivery_status = DeliveryStatus.PENDING
            self.persisted_response = None
            self.manifest = None
            self.queued_jobs.append(linked_job)
        self.events.append(f"link_turn:{turn_event_id}:{run_id}")
        return run_id, is_new_run

    async def complete_turn(self, event_id: str, target: SlackTurnStatus) -> bool:
        turn = self.turns[event_id]
        if turn.status is target:
            return False
        self.turns[event_id] = replace(
            turn,
            status=target,
            completed_at=datetime.now(UTC),
        )
        self.events.append(f"complete_turn:{event_id}:{target.value}")
        return True

    async def get_turn(self, event_id: str) -> SlackTurnRecord | None:
        return self.turns.get(event_id)

    async def get_latest_delivered_agent_response(
        self,
        team_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> AgentResponse | None:
        assert (team_id, channel_id, thread_ts) == ("T1", "C1", "1.0")
        self.events.append("load_latest_delivered_response")
        return self.latest_delivered_response

    async def claim_run(self, _run_id: object) -> RunClaim:
        if self.cancellation_requested:
            self.events.append("claim")
            return RunClaim(self.status, False, True)
        if self.status is RunStatus.QUEUED:
            self.status = RunStatus.RUNNING
        self.events.append("claim")
        return RunClaim(self.status, True, self.cancellation_requested)

    async def observe_run(self, _run_id: object) -> RunObservation:
        scripted_observations = self.run_observation_scripts.get(_run_id)
        if scripted_observations:
            observation = scripted_observations.pop(0)
            self.status = observation.status
            self.cancellation_requested = observation.cancellation_requested
            self.events.append(f"observe_run:{observation.status.value}")
            return observation
        return RunObservation(self.status, self.cancellation_requested)

    async def mark_succeeded(self, _run_id: object, response: AgentResponse) -> None:
        assert self.status is RunStatus.RUNNING
        assert response == self.persisted_response
        assert self.delivery_completed
        self.status = RunStatus.SUCCEEDED
        self.latest_delivered_response = response
        self.events.append("mark_succeeded")

    async def get_delivery(self, _run_id: object) -> DeliveryState:
        return DeliveryState(
            channel_id="C1",
            thread_ts="1.0",
            response_ts="2.0" if self.delivery_status is DeliveryStatus.DELIVERED else None,
            team_id="T1",
            user_id="U1",
            delivery_status=self.delivery_status,
        )

    async def get_delivery_manifest(self, _run_id: object) -> DeliveryManifest | None:
        return self.manifest

    async def request_cancellation(self, _run_id: object) -> RunObservation:
        if self.delivery_status not in {
            DeliveryStatus.DELIVERING,
            DeliveryStatus.DELIVERED,
            DeliveryStatus.CANCELLED,
        }:
            self.cancellation_requested = True
        return RunObservation(self.status, self.cancellation_requested)

    async def mark_cancelled(self, run_id: object) -> bool:
        assert self.cancellation_requested
        self.status = RunStatus.CANCELLED
        self.delivery_status = DeliveryStatus.CANCELLED
        self.cancelled_run_ids.add(run_id)
        self.events.append("mark_cancelled")
        return True

    async def mark_failed(self, _run_id: object, *, code: str, message: str) -> None:
        del message
        self.status = RunStatus.FAILED
        self.events.append(f"mark_failed:{code}")


class FakePublisher:
    def __init__(
        self,
        events: list[str],
        ledger: FakeWorkflowLedger,
        *,
        should_claim_delivery: bool = True,
        rejected_delivery_part: int | None = None,
    ) -> None:
        self.events = events
        self.ledger = ledger
        self.should_claim_delivery = should_claim_delivery
        self.rejected_delivery_part = rejected_delivery_part
        self.surface_timestamps: dict[object, str] = {}

    async def claim_progress_surface(self, run_id: object) -> ProgressSurfaceClaim:
        self.events.append("surface_claim")
        timestamp = self.surface_timestamps.get(run_id)
        if timestamp is not None:
            return ProgressSurfaceClaim(
                action=ProgressSurfaceAction.READY,
                timestamp=timestamp,
            )
        return ProgressSurfaceClaim(action=ProgressSurfaceAction.START)

    async def start_claimed_stream(self, _run_id: object) -> str:
        self.events.append("surface_start")
        return "2.0"

    async def finish_progress_surface(
        self,
        run_id: object,
        timestamp: str | None,
    ) -> str | None:
        self.events.append("surface_finish")
        if timestamp is not None:
            self.surface_timestamps[run_id] = timestamp
        return timestamp

    async def publish_progress(self, _run_id: object, event: ProgressEvent) -> bool:
        self.events.append(f"progress:{event.sequence}")
        return True

    async def prepare_delivery(self, _run_id: object, response: AgentResponse) -> PreparedDelivery:
        self.events.append("prepare")
        text = f"**Answer**\n\n{response.answer}"
        return PreparedDelivery(
            parts=(text,),
            content_hashes=(hashlib.sha256(text.encode()).hexdigest(),),
        )

    async def begin_delivery(self, _run_id: object) -> bool:
        self.events.append("claim_delivery")
        return self.should_claim_delivery

    async def publish_delivery_part(
        self,
        _run_id: object,
        _prepared: PreparedDelivery,
        part_number: int,
    ) -> str:
        if part_number == self.rejected_delivery_part:
            raise SlackDeliveryRejectedError("definitive test rejection")
        self.events.append(f"publish:{part_number}")
        return "2.0"

    async def complete_delivery(self, _run_id: object) -> None:
        self.events.append("complete_delivery")
        self.ledger.delivery_completed = True
        self.ledger.delivery_status = DeliveryStatus.DELIVERED

    async def publish_cancelled(self, _run_id: object) -> None:
        self.events.append("publish_cancelled")
        self.ledger.delivery_status = DeliveryStatus.CANCELLED

    async def publish_safe_error(self, _run_id: object) -> None:
        self.events.append("publish_safe_error")
        self.ledger.delivery_status = DeliveryStatus.FAILED

    async def publish_incomplete_delivery_notice(self, _run_id: object) -> None:
        self.events.append("publish_incomplete")
        self.ledger.delivery_status = DeliveryStatus.FAILED

    async def abandon_unreconciled_cancellation(self, _run_id: object) -> None:
        self.events.append("abandon_unreconciled_cancellation")
        self.ledger.delivery_status = DeliveryStatus.CANCELLED

    async def abandon_unreconciled_failure(self, _run_id: object) -> None:
        self.events.append("abandon_unreconciled_failure")
        self.ledger.delivery_status = DeliveryStatus.FAILED


class FakeResponderClassifier:
    def __init__(self, decision: ResponderDecision) -> None:
        self.decision = decision
        self.requests: list[ResponderClassificationRequest] = []

    async def classify(
        self,
        request: ResponderClassificationRequest,
    ) -> ResponderClassification:
        self.requests.append(request)
        return ResponderClassification(decision=self.decision)


class FakeInngestClient:
    def __init__(self) -> None:
        self.registrations: list[dict[str, object]] = []
        self.functions: list[Any] = []
        self.sent_events: list[Any] = []

    async def send(self, event: Any) -> list[str]:
        self.sent_events.append(event)
        return ["routed-question-event"]

    def create_function(self, **kwargs: object) -> Callable[[Any], Any]:
        self.registrations.append(kwargs)

        def decorate(function: Any) -> Any:
            function._fake_inngest_function_id = kwargs["fn_id"]
            self.functions.append(function)
            return function

        return decorate


class FakeStep:
    def __init__(
        self,
        *,
        lose_run_agent_acknowledgement: bool = False,
        trace: list[str] | None = None,
        sleep_callbacks: list[Callable[[], Awaitable[None]]] | None = None,
        failing_function_ids: set[str] | None = None,
    ) -> None:
        self._lose_run_agent_acknowledgement = lose_run_agent_acknowledgement
        self._trace = trace
        self._sleep_callbacks = list(sleep_callbacks or ())
        self._failing_function_ids = failing_function_ids or set()
        self.step_ids: list[str] = []
        self.invoked_function_ids: list[str] = []
        self.sleep_ids: list[str] = []

    async def run(
        self,
        step_id: str,
        function: Callable[..., Any],
        *args: Any,
    ) -> Any:
        self.step_ids.append(step_id)
        result = await function(*args)
        if step_id == "run-agent" and self._lose_run_agent_acknowledgement:
            raise RuntimeError("simulated lost Inngest step acknowledgement")
        return result

    async def invoke(
        self,
        step_id: str,
        *,
        function: Callable[[Any], Any],
        data: dict[str, Any],
        timeout: object | None = None,  # noqa: ASYNC109 - mirrors Inngest's API.
        v: object | None = None,
    ) -> Any:
        del timeout, v
        self.step_ids.append(step_id)
        function_id = str(cast(Any, function)._fake_inngest_function_id)
        self.invoked_function_ids.append(function_id)
        if self._trace is not None:
            self._trace.append(f"invoke:{function_id}")
        if function_id in self._failing_function_ids:
            raise RuntimeError(f"simulated {function_id} failure")
        context = SimpleNamespace(
            event=SimpleNamespace(
                name="slack/question.ready",
                id=f"invoke:{step_id}",
                ts=1,
                data=data,
            ),
            step=self,
        )
        return await function(cast(Any, context))

    async def sleep(
        self,
        step_id: str,
        duration: object,
    ) -> None:
        del duration
        self.step_ids.append(step_id)
        self.sleep_ids.append(step_id)
        if self._trace is not None:
            self._trace.append(f"sleep:{step_id}")
        if self._sleep_callbacks:
            await self._sleep_callbacks.pop(0)()


class FakeContext:
    def __init__(
        self,
        job: QuestionJob,
        *,
        lose_run_agent_acknowledgement: bool = False,
        step: FakeStep | None = None,
    ) -> None:
        self.event = SimpleNamespace(
            name="slack/question.received",
            id="InngestEvent1",
            ts=1,
            data=job.model_dump(mode="json"),
        )
        self.step = step or FakeStep(lose_run_agent_acknowledgement=lose_run_agent_acknowledgement)


class FakeFollowUpContext:
    def __init__(
        self,
        candidate: FollowUpCandidateJob,
        *,
        step: FakeStep | None = None,
    ) -> None:
        self.event = SimpleNamespace(
            name="slack/follow_up.candidate",
            id="InngestFollowUp1",
            ts=2,
            data=candidate.model_dump(mode="json"),
        )
        self.step = step or FakeStep()


def _registered_functions(client: FakeInngestClient) -> dict[str, Any]:
    functions = {
        str(registration["fn_id"]): function
        for registration, function in zip(
            client.registrations,
            client.functions,
            strict=True,
        )
    }
    assert len(functions) == len(client.functions)
    return functions


def _function_by_id(client: FakeInngestClient, function_id: str) -> Any:
    functions = _registered_functions(client)
    assert function_id in functions
    return functions[function_id]


def _registration_by_id(client: FakeInngestClient, function_id: str) -> dict[str, object]:
    registration = next(
        (candidate for candidate in client.registrations if candidate.get("fn_id") == function_id),
        None,
    )
    assert registration is not None
    return registration


def _job(
    *,
    event_id: str = "Ev1",
    message_ts: str = "1.0",
    question: str = "What changed?",
) -> QuestionJob:
    return QuestionJob(
        event_id=event_id,
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts=message_ts,
        thread_ts="1.0",
        question=question,
    )


def _follow_up(
    message_text: str,
    *,
    event_id: str = "EvFollowUp1",
    message_ts: str = "2.0",
) -> FollowUpCandidateJob:
    return FollowUpCandidateJob(
        event_id=event_id,
        team_id="T1",
        channel_id="C1",
        user_id="U2",
        message_ts=message_ts,
        thread_ts="1.0",
        message_text=message_text,
    )


async def test_lost_run_step_ack_reuses_result_and_succeeds_only_after_delivery() -> None:
    events: list[str] = []
    response = AgentResponse(answer="Grounded answer", model_call_count=4)
    processor = FakeStreamingProcessor(response, events)
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    job = _job()

    def processor_provider() -> StreamingQuestionProcessor:
        return cast(StreamingQuestionProcessor, processor)

    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=processor_provider,
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    process_question = _function_by_id(client, "process-slack-question")

    with pytest.raises(RuntimeError, match="lost Inngest step acknowledgement"):
        await process_question(cast(Any, FakeContext(job, lose_run_agent_acknowledgement=True)))

    assert ledger.status is RunStatus.RUNNING
    assert ledger.persisted_response == response
    assert not any(event.startswith("publish:") for event in events)

    result = await process_question(cast(Any, FakeContext(job)))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
    assert processor.call_count == 1
    # The workflow mutates the fake through a protocol-typed callback, which mypy cannot see.
    assert cast(RunStatus, ledger.status) is RunStatus.SUCCEEDED
    assert events.count("graph") == 1
    assert events.index("publish:1") < events.index("complete_delivery")
    assert events.index("complete_delivery") < events.index("mark_succeeded")


async def test_retry_recovers_run_after_delivery_ack_was_lost() -> None:
    events: list[str] = []
    response = AgentResponse(answer="Grounded answer", model_call_count=4)
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.persisted_response = response
    ledger.delivery_completed = True
    ledger.delivery_status = DeliveryStatus.DELIVERED
    publisher = FakePublisher(events, ledger, should_claim_delivery=False)
    job = _job()

    def fail_if_processor_is_requested() -> StreamingQuestionProcessor:
        raise AssertionError("retry must reuse the persisted response")

    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=fail_if_processor_is_requested,
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    process_question = _function_by_id(client, "process-slack-question")

    result = await process_question(cast(Any, FakeContext(job)))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
    assert ledger.status is RunStatus.SUCCEEDED
    assert events.count("mark_succeeded") == 1
    assert not any(event.startswith("publish:") for event in events)


async def test_primary_function_finalizes_a_persisted_stop_without_cancel_event() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.cancellation_requested = True
    publisher = FakePublisher(events, ledger)
    job = _job()
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="must not run"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    process_question = _function_by_id(client, "process-slack-question")

    result = await process_question(cast(Any, FakeContext(job)))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "cancelled"}
    assert ledger.status is RunStatus.CANCELLED
    assert events == ["claim", "publish_cancelled", "mark_cancelled"]


async def test_progress_initializer_opens_surface_without_model_function_capacity() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )

    context = FakeContext(_job())
    initialize_progress = _function_by_id(client, "initialize-slack-progress")
    result = await initialize_progress(cast(Any, context))

    assert result["status"] == "initialized"
    assert result["slack_message_ts"] == "2.0"
    assert events == ["surface_claim", "surface_start", "surface_finish"]
    assert context.step.step_ids == [
        "claim-progress-surface",
        "start-slack-stream",
        "acknowledge-slack-stream",
    ]


def _register_execution_functions(
    *,
    ledger: FakeWorkflowLedger,
    publisher: FakePublisher,
    processor: FakeStreamingProcessor,
    classifier: FakeResponderClassifier | None,
) -> FakeInngestClient:
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(StreamingQuestionProcessor, processor),
        responder_classifier=(
            cast(ResponderClassifier, classifier) if classifier is not None else None
        ),
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    return client


async def test_explicit_mention_uses_shared_router_and_bypasses_classifier() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Explicit answer"), events)
    classifier = FakeResponderClassifier(ResponderDecision.STAY_SILENT)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=classifier,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    context = FakeContext(_job(), step=FakeStep(trace=events))

    result = await route_turn(cast(Any, context))

    turn = await ledger.get_turn("Ev1")
    assert turn is not None
    assert turn.kind is SlackTurnKind.EXPLICIT_MENTION
    assert turn.status is SlackTurnStatus.ROUTED
    assert turn.agent_run_id is not None
    assert result["agent_run_id"] == str(turn.agent_run_id)
    assert result["status"] == "succeeded"
    assert classifier.requests == []
    assert context.step.invoked_function_ids == [
        "initialize-slack-progress",
        "process-slack-question",
    ]
    assert events.index(f"link_turn:Ev1:{turn.agent_run_id}") < events.index(
        "invoke:initialize-slack-progress"
    )
    assert client.sent_events == []


async def test_progress_initializer_failure_does_not_block_processing_or_routing() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Grounded answer"), events)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=None,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    step = FakeStep(
        trace=events,
        failing_function_ids={"initialize-slack-progress"},
    )

    result = await route_turn(cast(Any, FakeContext(_job(), step=step)))

    turn = await ledger.get_turn("Ev1")
    assert turn is not None
    assert turn.status is SlackTurnStatus.ROUTED
    assert result == {
        "agent_run_id": str(turn.agent_run_id),
        "event_id": "Ev1",
        "status": "succeeded",
    }
    assert processor.call_count == 1
    assert step.invoked_function_ids == [
        "initialize-slack-progress",
        "process-slack-question",
    ]
    assert events.index("invoke:initialize-slack-progress") < events.index(
        "invoke:process-slack-question"
    )


async def test_one_word_follow_up_invokes_classifier_progress_and_process_in_order() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.latest_delivered_response = AgentResponse(
        answer="Which customer?",
        disposition=QuestionDisposition.NEEDS_CLARIFICATION,
    )
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Acme answer"), events)
    classifier = FakeResponderClassifier(ResponderDecision.RESPOND)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=classifier,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    candidate = _follow_up("Acme")
    context = FakeFollowUpContext(candidate, step=FakeStep(trace=events))

    result = await route_turn(cast(Any, context))

    turn = await ledger.get_turn("EvFollowUp1")
    assert turn is not None
    assert turn.kind is SlackTurnKind.FOLLOW_UP
    assert turn.status is SlackTurnStatus.ROUTED
    assert turn.agent_run_id == candidate.candidate_id
    assert result["agent_run_id"] == str(turn.agent_run_id)
    assert len(classifier.requests) == 1
    assert classifier.requests[0].message_text == "Acme"
    assert classifier.requests[0].last_agent_clarification_question == "Which customer?"
    assert classifier.requests[0].last_agent_response == "Which customer?"
    assert len(ledger.queued_jobs) == 1
    assert ledger.queued_jobs[0].agent_run_id == candidate.candidate_id
    assert ledger.queued_jobs[0].question == "Acme"
    assert context.step.invoked_function_ids == [
        "classify-slack-follow-up",
        "initialize-slack-progress",
        "process-slack-question",
    ]
    link_event = f"link_turn:EvFollowUp1:{turn.agent_run_id}"
    assert events.index(link_event) < events.index("invoke:initialize-slack-progress")
    assert events.index("invoke:initialize-slack-progress") < events.index(
        "invoke:process-slack-question"
    )
    assert client.sent_events == []


async def test_suppressed_follow_up_completes_turn_without_agent_children() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.latest_delivered_response = AgentResponse(
        answer="Which customer?",
        disposition=QuestionDisposition.NEEDS_CLARIFICATION,
    )
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="must not run"), events)
    classifier = FakeResponderClassifier(ResponderDecision.STAY_SILENT)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=classifier,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    context = FakeFollowUpContext(
        _follow_up("<@U3> can you send that later?"),
        step=FakeStep(trace=events),
    )

    result = await route_turn(cast(Any, context))

    turn = await ledger.get_turn("EvFollowUp1")
    assert turn is not None
    assert turn.status is SlackTurnStatus.SUPPRESSED
    assert turn.agent_run_id is None
    assert result["status"] == "classifier_stay_silent"
    assert context.step.invoked_function_ids == ["classify-slack-follow-up"]
    assert processor.call_count == 0
    assert ledger.queued_jobs == []
    assert client.sent_events == []


async def test_explicit_follow_up_recovers_suppressed_human_thread_context() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Grounded answer"), events)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=None,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    suppressed = _follow_up(
        "do they have any other old patch windows?",
        event_id="EvSuppressed",
        message_ts="2.0",
    )
    await ledger.ensure_turn(
        event_id=suppressed.event_id,
        team_id=suppressed.team_id,
        channel_id=suppressed.channel_id,
        user_id=suppressed.user_id,
        message_ts=suppressed.message_ts,
        thread_ts=suppressed.thread_ts,
        message_text=suppressed.message_text,
        kind=SlackTurnKind.FOLLOW_UP,
    )
    await ledger.claim_turn(suppressed.event_id)
    await ledger.complete_turn(suppressed.event_id, SlackTurnStatus.SUPPRESSED)

    await route_turn(
        cast(Any, FakeContext(_job(event_id="EvExplicit", message_ts="3.0", question="??")))
    )

    assert processor.questions == [
        "Earlier unanswered human messages in this Slack thread (untrusted context):\n"
        "do they have any other old patch windows?\n\nCurrent explicit message:\n??"
    ]


async def test_explicit_follow_up_recovers_cancelled_explicit_mention_context() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Grounded answer"), events)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=None,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    cancelled_job = _job(
        event_id="EvCancelled",
        message_ts="2.0",
        question="What is the Canada approval-bypass pattern?",
    )
    await ledger.ensure_turn(
        event_id=cancelled_job.event_id,
        team_id=cancelled_job.team_id,
        channel_id=cancelled_job.channel_id,
        user_id=cancelled_job.user_id,
        message_ts=cancelled_job.message_ts,
        thread_ts=cancelled_job.thread_ts,
        message_text=cancelled_job.question,
        kind=SlackTurnKind.EXPLICIT_MENTION,
    )
    await ledger.claim_turn(cancelled_job.event_id)
    await ledger.create_queued_for_turn(cancelled_job, cancelled_job.event_id)
    ledger.cancellation_requested = True
    await ledger.mark_cancelled(cancelled_job.agent_run_id)
    await ledger.complete_turn(cancelled_job.event_id, SlackTurnStatus.ROUTED)

    await route_turn(
        cast(
            Any,
            FakeContext(
                _job(
                    event_id="EvContinue",
                    message_ts="3.0",
                    question="please continue working on the request",
                )
            ),
        )
    )

    assert processor.questions == [
        "Earlier unanswered human messages in this Slack thread (untrusted context):\n"
        "What is the Canada approval-bypass pattern?\n\n"
        "Current explicit message:\n"
        "please continue working on the request"
    ]


async def test_follow_up_without_visible_agent_response_is_suppressed_without_child() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="must not run"), events)
    classifier = FakeResponderClassifier(ResponderDecision.RESPOND)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=classifier,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    context = FakeFollowUpContext(_follow_up("why?"))

    result = await route_turn(cast(Any, context))

    turn = await ledger.get_turn("EvFollowUp1")
    assert turn is not None
    assert turn.status is SlackTurnStatus.SUPPRESSED
    assert result["status"] == "not_agent_owned"
    assert classifier.requests == []
    assert context.step.invoked_function_ids == []
    assert processor.call_count == 0
    assert client.sent_events == []


async def test_non_head_turn_sleeps_until_a_later_claim_is_permitted() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Later answer"), events)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=None,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    await ledger.ensure_turn(
        event_id="Earlier",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="1.5",
        thread_ts="1.0",
        kind=SlackTurnKind.EXPLICIT_MENTION,
    )
    ledger.script_turn_claims("Later", False, True)

    async def release_earlier_turn() -> None:
        claim = await ledger.claim_turn("Earlier")
        assert claim.should_process
        await ledger.complete_turn("Earlier", SlackTurnStatus.SUPPRESSED)

    step = FakeStep(trace=events, sleep_callbacks=[release_earlier_turn])
    context = FakeContext(
        _job(event_id="Later", message_ts="2.0", question="Later question"),
        step=step,
    )

    await route_turn(cast(Any, context))

    assert step.sleep_ids == ["wait-for-causal-turn-1"]
    assert step.step_ids.index("claim-turn-1") < step.step_ids.index("wait-for-causal-turn-1")
    assert step.step_ids.index("wait-for-causal-turn-1") < step.step_ids.index("claim-turn-2")
    assert events.index("claim_turn:Later:claimed") < events.index(
        "invoke:initialize-slack-progress"
    )
    assert step.invoked_function_ids == [
        "initialize-slack-progress",
        "process-slack-question",
    ]


async def test_follow_up_head_finishes_before_later_explicit_children_are_invoked() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.latest_delivered_response = AgentResponse(
        answer="Which customer?",
        disposition=QuestionDisposition.NEEDS_CLARIFICATION,
    )
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Grounded answer"), events)
    classifier = FakeResponderClassifier(ResponderDecision.RESPOND)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=classifier,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    follow_up = _follow_up("Acme", event_id="EarlierFollowUp", message_ts="2.0")
    await ledger.ensure_turn(
        event_id=follow_up.event_id,
        team_id=follow_up.team_id,
        channel_id=follow_up.channel_id,
        user_id=follow_up.user_id,
        message_ts=follow_up.message_ts,
        thread_ts=follow_up.thread_ts,
        kind=SlackTurnKind.FOLLOW_UP,
    )

    async def route_earlier_follow_up() -> None:
        earlier_step = FakeStep(trace=events)
        await route_turn(cast(Any, FakeFollowUpContext(follow_up, step=earlier_step)))

    later_step = FakeStep(trace=events, sleep_callbacks=[route_earlier_follow_up])
    later_context = FakeContext(
        _job(event_id="LaterExplicit", message_ts="3.0", question="What about timing?"),
        step=later_step,
    )

    await route_turn(cast(Any, later_context))

    first_progress = events.index("invoke:initialize-slack-progress")
    first_process = events.index("invoke:process-slack-question")
    earlier_complete = events.index("complete_turn:EarlierFollowUp:routed")
    later_claim = events.index("claim_turn:LaterExplicit:claimed")
    progress_positions = [
        index for index, event in enumerate(events) if event == "invoke:initialize-slack-progress"
    ]
    process_positions = [
        index for index, event in enumerate(events) if event == "invoke:process-slack-question"
    ]
    assert first_progress < first_process < earlier_complete < later_claim
    assert later_claim < progress_positions[1] < process_positions[1]
    assert len(classifier.requests) == 1
    assert classifier.requests[0].message_text == "Acme"
    assert processor.call_count == 2
    assert client.sent_events == []


async def test_terminal_turn_replay_invokes_no_child_function() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    processor = FakeStreamingProcessor(AgentResponse(answer="Grounded answer"), events)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=processor,
        classifier=None,
    )
    route_turn = _function_by_id(client, "route-slack-turn")
    job = _job()
    await route_turn(cast(Any, FakeContext(job)))
    replay_context = FakeContext(job)

    result = await route_turn(cast(Any, replay_context))

    assert result["status"] == "routed"
    assert replay_context.step.invoked_function_ids == []
    assert processor.call_count == 1


async def test_system_cancellation_is_reported_as_neutral_failure() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    job = _job()
    context = SimpleNamespace(
        event=SimpleNamespace(
            name=FUNCTION_CANCELLED_EVENT,
            id="SystemCancel1",
            ts=1,
            data={"event": {"data": job.model_dump(mode="json")}},
        ),
        step=FakeStep(),
    )

    cleanup_cancelled = _function_by_id(client, "cleanup-cancelled-slack-question")
    result = await cleanup_cancelled(cast(Any, context))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "failed"}
    assert "publish_safe_error" in events
    assert "publish_cancelled" not in events
    assert "mark_failed:inngest_function_cancelled" in events


async def test_failure_cleanup_recovers_fully_acknowledged_answer() -> None:
    events: list[str] = []
    response = AgentResponse(answer="Grounded answer")
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.persisted_response = response
    ledger.delivery_status = DeliveryStatus.DELIVERING
    ledger.manifest = DeliveryManifest(
        version=1,
        manifest_hash="a" * 64,
        parts=(
            DeliveryPartState(
                part_number=1,
                content_hash="b" * 64,
                slack_message_ts="2.0",
                acknowledged_at=datetime.now(UTC),
            ),
        ),
    )
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(response, events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    job = _job()
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="Failure1",
            ts=1,
            data={"event": {"data": job.model_dump(mode="json")}},
        ),
        step=FakeStep(),
    )

    cleanup_failed = _function_by_id(client, "cleanup-failed-slack-question")
    result = await cleanup_failed(cast(Any, context))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
    assert ledger.status is RunStatus.SUCCEEDED
    assert events.index("complete_delivery") < events.index("mark_succeeded")
    assert "publish_safe_error" not in events


async def test_failure_cleanup_retries_unacknowledged_persisted_answer() -> None:
    events: list[str] = []
    response = AgentResponse(answer="Grounded answer")
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.persisted_response = response
    ledger.delivery_status = DeliveryStatus.DELIVERING
    ledger.manifest = DeliveryManifest(
        version=1,
        manifest_hash="a" * 64,
        parts=(
            DeliveryPartState(
                part_number=1,
                content_hash="b" * 64,
                slack_message_ts=None,
                acknowledged_at=None,
            ),
        ),
    )
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(response, events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    job = _job()
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="FailureUnacknowledged1",
            ts=1,
            data={"event": {"data": job.model_dump(mode="json")}},
        ),
        step=FakeStep(),
    )

    cleanup_failed = _function_by_id(client, "cleanup-failed-slack-question")
    result = await cleanup_failed(cast(Any, context))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
    assert ledger.status is RunStatus.SUCCEEDED
    assert "publish:1" in events
    assert events.index("publish:1") < events.index("complete_delivery")
    assert events.index("complete_delivery") < events.index("mark_succeeded")
    assert "publish_safe_error" not in events


async def test_definitive_recovery_rejection_falls_through_to_safe_error() -> None:
    events: list[str] = []
    response = AgentResponse(answer="Grounded answer")
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.persisted_response = response
    ledger.delivery_status = DeliveryStatus.DELIVERING
    ledger.manifest = DeliveryManifest(
        version=1,
        manifest_hash="a" * 64,
        parts=(
            DeliveryPartState(
                part_number=1,
                content_hash="b" * 64,
                slack_message_ts=None,
                acknowledged_at=None,
            ),
        ),
    )
    publisher = FakePublisher(events, ledger, rejected_delivery_part=1)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(response, events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    job = _job()
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="FailureRejected1",
            ts=1,
            data={"event": {"data": job.model_dump(mode="json")}},
        ),
        step=FakeStep(),
    )

    cleanup_failed = _function_by_id(client, "cleanup-failed-slack-question")
    result = await cleanup_failed(cast(Any, context))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "failed"}
    assert ledger.status is RunStatus.FAILED
    assert "publish_safe_error" in events
    assert "mark_failed:inngest_retries_exhausted" in events


async def test_rejected_stop_does_not_cancel_delivery_winner() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.delivery_status = DeliveryStatus.DELIVERING
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    cancellation = QuestionCancellationJob(
        event_id="EvStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="3.0",
        streaming_message_ts=("2.0",),
        agent_run_id=_job().agent_run_id,
        cancellation_accepted=False,
    )
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="slack/question.cancelled",
            id="Stop1",
            ts=1,
            data=cancellation.model_dump(mode="json"),
        ),
        step=FakeStep(),
    )

    cleanup_cancelled = _function_by_id(client, "cleanup-cancelled-slack-question")
    result = await cleanup_cancelled(cast(Any, context))

    assert result["status"] == "cancellation_rejected"
    assert "publish_cancelled" not in events


async def test_stop_without_a_linked_run_does_not_rewrite_session_status() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    cancellation = QuestionCancellationJob(
        event_id="EvStaleStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="3.0",
        streaming_message_ts=(),
    )
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="slack/question.cancelled",
            id="StaleStop1",
            ts=1,
            data=cancellation.model_dump(mode="json"),
        ),
        step=FakeStep(),
    )

    cleanup_cancelled = _function_by_id(client, "cleanup-cancelled-slack-question")
    result = await cleanup_cancelled(cast(Any, context))

    assert result == {"agent_run_id": None, "status": "no_active_run"}
    assert events == []
    assert context.step.step_ids == []


async def test_delayed_stop_does_not_overwrite_a_succeeded_clarification_session() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.SUCCEEDED
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    job = _job()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(
                AgentResponse(
                    answer="Which customer?",
                    disposition=QuestionDisposition.NEEDS_CLARIFICATION,
                ),
                events,
            ),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    cancellation = QuestionCancellationJob(
        event_id="EvDelayedStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="3.0",
        streaming_message_ts=("2.0",),
        agent_run_id=job.agent_run_id,
        cancellation_accepted=False,
    )
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="slack/question.cancelled",
            id="DelayedStop1",
            ts=1,
            data=cancellation.model_dump(mode="json"),
        ),
        step=FakeStep(),
    )

    cleanup_cancelled = _function_by_id(client, "cleanup-cancelled-slack-question")
    result = await cleanup_cancelled(cast(Any, context))

    assert result == {"agent_run_id": str(job.agent_run_id), "status": "succeeded"}
    assert events == []
    assert context.step.step_ids == []


async def test_failed_router_marks_an_unlinked_processing_turn_failed() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=FakeStreamingProcessor(AgentResponse(answer="unused"), events),
        classifier=None,
    )
    job = _job(event_id="UnlinkedTurn")
    await ledger.ensure_turn(
        event_id=job.event_id,
        team_id=job.team_id,
        channel_id=job.channel_id,
        user_id=job.user_id,
        message_ts=job.message_ts,
        thread_ts=job.thread_ts,
        kind=SlackTurnKind.EXPLICIT_MENTION,
    )
    claim = await ledger.claim_turn(job.event_id)
    assert claim.turn.status is SlackTurnStatus.PROCESSING
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="FailedRouterUnlinked1",
            ts=1,
            data={
                "event": {
                    "name": "slack/question.received",
                    "data": job.model_dump(mode="json"),
                }
            },
        ),
        step=FakeStep(),
    )

    cleanup_failed_turn = _function_by_id(client, "cleanup-failed-slack-turn")
    result = await cleanup_failed_turn(cast(Any, context))

    turn = await ledger.get_turn(job.event_id)
    assert turn is not None
    assert turn.status is SlackTurnStatus.FAILED
    assert turn.agent_run_id is None
    assert result == {"event_id": job.event_id, "status": "failed"}
    assert context.step.invoked_function_ids == []


async def test_failed_router_rechecks_a_blocked_pending_turn_after_sleep() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=FakeStreamingProcessor(AgentResponse(answer="unused"), events),
        classifier=None,
    )
    earlier = _job(event_id="EarlierTurn", message_ts="1.0")
    later = _job(event_id="LaterFailedTurn", message_ts="2.0")
    for job in (earlier, later):
        await ledger.ensure_turn(
            event_id=job.event_id,
            team_id=job.team_id,
            channel_id=job.channel_id,
            user_id=job.user_id,
            message_ts=job.message_ts,
            thread_ts=job.thread_ts,
            kind=SlackTurnKind.EXPLICIT_MENTION,
        )
    earlier_claim = await ledger.claim_turn(earlier.event_id)
    assert earlier_claim.should_process

    async def release_earlier_turn() -> None:
        await ledger.complete_turn(earlier.event_id, SlackTurnStatus.SUPPRESSED)

    step = FakeStep(sleep_callbacks=[release_earlier_turn])
    context = SimpleNamespace(
        event=SimpleNamespace(
            name=FUNCTION_CANCELLED_EVENT,
            id="CancelledBlockedRouter1",
            ts=1,
            data={
                "event": {
                    "name": "slack/question.received",
                    "data": later.model_dump(mode="json"),
                }
            },
        ),
        step=step,
    )

    cleanup_failed_turn = _function_by_id(client, "cleanup-failed-slack-turn")
    result = await cleanup_failed_turn(cast(Any, context))

    turn = await ledger.get_turn(later.event_id)
    assert turn is not None
    assert turn.status is SlackTurnStatus.FAILED
    assert result == {"event_id": later.event_id, "status": "failed"}
    assert step.sleep_ids == ["wait-to-clean-failed-turn-1"]
    assert "claim-failed-turn-1" in step.step_ids
    assert "claim-failed-turn-2" in step.step_ids


async def test_failed_router_keeps_linked_turn_processing_until_run_is_terminal() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=FakeStreamingProcessor(AgentResponse(answer="unused"), events),
        classifier=None,
    )
    job = _job(event_id="LinkedTurn")
    await ledger.ensure_turn(
        event_id=job.event_id,
        team_id=job.team_id,
        channel_id=job.channel_id,
        user_id=job.user_id,
        message_ts=job.message_ts,
        thread_ts=job.thread_ts,
        kind=SlackTurnKind.EXPLICIT_MENTION,
    )
    await ledger.claim_turn(job.event_id)
    await ledger.create_queued_for_turn(job, job.event_id)
    ledger.status = RunStatus.RUNNING
    ledger.script_run_observations(
        job.agent_run_id,
        RunObservation(RunStatus.RUNNING, False),
        RunObservation(RunStatus.SUCCEEDED, False),
    )
    statuses_during_wait: list[SlackTurnStatus] = []

    async def record_turn_status_during_wait() -> None:
        turn = await ledger.get_turn(job.event_id)
        assert turn is not None
        statuses_during_wait.append(turn.status)

    step = FakeStep(trace=events, sleep_callbacks=[record_turn_status_during_wait])
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="FailedRouterLinked1",
            ts=1,
            data={
                "event": {
                    "name": "slack/question.received",
                    "data": job.model_dump(mode="json"),
                }
            },
        ),
        step=step,
    )

    cleanup_failed_turn = _function_by_id(client, "cleanup-failed-slack-turn")
    result = await cleanup_failed_turn(cast(Any, context))

    turn = await ledger.get_turn(job.event_id)
    assert turn is not None
    assert statuses_during_wait == [SlackTurnStatus.PROCESSING]
    assert step.sleep_ids == ["wait-for-linked-run-cleanup-1"]
    assert turn.status is SlackTurnStatus.ROUTED
    assert result == {
        "agent_run_id": str(job.agent_run_id),
        "event_id": job.event_id,
        "status": "succeeded",
    }
    assert events.index("observe_run:running") < events.index("sleep:wait-for-linked-run-cleanup-1")
    assert events.index("observe_run:succeeded") < events.index(
        f"complete_turn:{job.event_id}:routed"
    )


async def test_unreconciled_turn_cleanup_fails_active_run_before_releasing_queue() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)
    client = _register_execution_functions(
        ledger=ledger,
        publisher=publisher,
        processor=FakeStreamingProcessor(AgentResponse(answer="unused"), events),
        classifier=None,
    )
    job = _job(event_id="UnreconciledTurn")
    await ledger.ensure_turn(
        event_id=job.event_id,
        team_id=job.team_id,
        channel_id=job.channel_id,
        user_id=job.user_id,
        message_ts=job.message_ts,
        thread_ts=job.thread_ts,
        kind=SlackTurnKind.EXPLICIT_MENTION,
    )
    await ledger.claim_turn(job.event_id)
    await ledger.create_queued_for_turn(job, job.event_id)
    ledger.status = RunStatus.RUNNING
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="UnreconciledTurnCleanup1",
            ts=1,
            data={
                "function_id": TURN_FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
                "event": {
                    "data": {
                        "event": {
                            "name": "slack/question.received",
                            "data": job.model_dump(mode="json"),
                        }
                    }
                },
            },
        ),
        step=FakeStep(),
    )
    event_count_before_cleanup = len(events)

    finalize_unreconciled = _function_by_id(client, "finalize-unreconciled-slack-run")
    result = await finalize_unreconciled(cast(Any, context))

    turn = await ledger.get_turn(job.event_id)
    assert turn is not None
    assert ledger.status is RunStatus.FAILED
    assert turn.status is SlackTurnStatus.ROUTED
    assert result == {
        "agent_run_id": str(job.agent_run_id),
        "event_id": job.event_id,
        "status": "failed",
    }
    assert events[event_count_before_cleanup:] == [
        "abandon_unreconciled_failure",
        "mark_failed:turn_reconciliation_exhausted",
        f"complete_turn:{job.event_id}:routed",
    ]


async def test_failed_cancellation_cleanup_releases_run_without_duplicate_notice() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.cancellation_requested = True
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    job = _job()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    cancellation = QuestionCancellationJob(
        event_id="EvUnreconciledStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="3.0",
        streaming_message_ts=("2.0",),
        agent_run_id=job.agent_run_id,
        cancellation_accepted=True,
    )
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="CleanupFailure1",
            ts=1,
            data={
                "function_id": CANCELLATION_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
                "event": {"data": cancellation.model_dump(mode="json")},
            },
        ),
        step=FakeStep(),
    )

    finalize_unreconciled = _function_by_id(client, "finalize-unreconciled-slack-run")
    result = await finalize_unreconciled(cast(Any, context))

    assert result == {
        "agent_run_id": str(job.agent_run_id),
        "status": "cancelled_unconfirmed",
    }
    assert ledger.status is RunStatus.CANCELLED
    assert events == ["abandon_unreconciled_cancellation", "mark_cancelled"]


async def test_failed_failure_cleanup_releases_uncertain_run_without_another_post() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    ledger.delivery_status = DeliveryStatus.DELIVERING
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    job = _job()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="FailureCleanupFailure1",
            ts=1,
            data={
                "function_id": FAILURE_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
                "event": {
                    "data": {"event": {"data": job.model_dump(mode="json")}},
                },
            },
        ),
        step=FakeStep(),
    )

    finalize_unreconciled = _function_by_id(client, "finalize-unreconciled-slack-run")
    result = await finalize_unreconciled(cast(Any, context))

    assert result == {
        "agent_run_id": str(job.agent_run_id),
        "status": "failed_unconfirmed",
    }
    assert ledger.status is RunStatus.FAILED
    assert events == [
        "load",
        "abandon_unreconciled_failure",
        "mark_failed:slack_delivery_reconciliation_exhausted",
    ]


async def test_failed_system_cancellation_cleanup_releases_run_as_failure() -> None:
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    ledger.status = RunStatus.RUNNING
    publisher = FakePublisher(events, ledger)
    client = FakeInngestClient()
    job = _job()
    create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )
    context = SimpleNamespace(
        event=SimpleNamespace(
            name="inngest/function.failed",
            id="SystemCancellationCleanupFailure1",
            ts=1,
            data={
                "function_id": CANCELLATION_CLEANUP_FULLY_QUALIFIED_FUNCTION_ID,
                "event": {
                    "data": {"event": {"data": job.model_dump(mode="json")}},
                },
            },
        ),
        step=FakeStep(),
    )

    finalize_unreconciled = _function_by_id(client, "finalize-unreconciled-slack-run")
    result = await finalize_unreconciled(cast(Any, context))

    assert result == {
        "agent_run_id": str(job.agent_run_id),
        "status": "failed_unconfirmed",
    }
    assert ledger.status is RunStatus.FAILED
    assert events == [
        "abandon_unreconciled_failure",
        "mark_failed:system_cancellation_reconciliation_exhausted",
    ]


def test_question_function_has_cancellation_timeouts_and_cleanup_functions() -> None:
    client = FakeInngestClient()
    events: list[str] = []
    ledger = FakeWorkflowLedger(events)
    publisher = FakePublisher(events, ledger)

    functions = create_question_functions(
        cast(Any, client),
        processor_provider=lambda: cast(
            StreamingQuestionProcessor,
            FakeStreamingProcessor(AgentResponse(answer="ok"), events),
        ),
        responder_classifier=None,
        ledger=cast(RunLedger, ledger),
        publisher=cast(SlackPublisher, publisher),
    )

    assert len(functions) == 8
    assert set(_registered_functions(client)) == {
        "process-slack-question",
        "initialize-slack-progress",
        "classify-slack-follow-up",
        "route-slack-turn",
        "cleanup-failed-slack-turn",
        "cleanup-failed-slack-question",
        "cleanup-cancelled-slack-question",
        "finalize-unreconciled-slack-run",
    }
    primary = _registration_by_id(client, "process-slack-question")
    assert primary["cancel"]
    assert primary["timeouts"]
    assert primary["idempotency"] == "event.data.event_id"
    primary_concurrency = cast(list[Any], primary["concurrency"])
    model_concurrency = next(
        constraint for constraint in primary_concurrency if constraint.key == '"openai"'
    )
    router = _registration_by_id(client, "route-slack-turn")
    router_concurrency = cast(list[Any], router["concurrency"])
    conversation_concurrency = next(
        constraint
        for constraint in router_concurrency
        if constraint.key == "event.data.conversation_id"
    )
    assert conversation_concurrency.limit == 1
    assert conversation_concurrency.scope == "env"
    progress = _registration_by_id(client, "initialize-slack-progress")
    assert progress["idempotency"] == "event.data.event_id"
    progress_concurrency = cast(list[Any], progress["concurrency"])
    assert any(constraint.key == '"slack_progress"' for constraint in progress_concurrency)
    classifier = _registration_by_id(client, "classify-slack-follow-up")
    classifier_concurrency = cast(list[Any], classifier["concurrency"])
    assert model_concurrency in classifier_concurrency
    turn_cleanup = _registration_by_id(client, "cleanup-failed-slack-turn")
    assert turn_cleanup["timeouts"]
    turn_cleanup_triggers = cast(list[Any], turn_cleanup["trigger"])
    assert {trigger.event for trigger in turn_cleanup_triggers} == {
        FUNCTION_FAILED_EVENT,
        FUNCTION_CANCELLED_EVENT,
    }
    terminal_reconciler = _registration_by_id(client, "finalize-unreconciled-slack-run")
    assert terminal_reconciler["timeouts"]
    terminal_reconciler_triggers = cast(list[Any], terminal_reconciler["trigger"])
    assert {trigger.event for trigger in terminal_reconciler_triggers} == {
        FUNCTION_FAILED_EVENT,
        FUNCTION_CANCELLED_EVENT,
    }
