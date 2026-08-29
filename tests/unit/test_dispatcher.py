from typing import Any
from uuid import UUID, uuid4

import inngest

from knowledge_assistant.execution.dispatcher import (
    FOLLOW_UP_CANDIDATE_EVENT,
    QUESTION_CANCELLED_EVENT,
    InngestAgentSessionStopHandoff,
    InngestFollowUpCandidateDispatcher,
    InngestQuestionDispatcher,
)
from knowledge_assistant.execution.models import (
    AgentSessionStopRequest,
    FollowUpCandidateJob,
    QuestionJob,
)
from knowledge_assistant.persistence.repositories import CancellationClaim


class FakeInngest:
    def __init__(self) -> None:
        self.event: inngest.Event | None = None

    async def send(self, event: inngest.Event) -> list[str]:
        self.event = event
        return ["event-result"]


class FakeCancellationLedger:
    def __init__(self, claim: CancellationClaim) -> None:
        self.claim = claim
        self.claimed_events: list[str] = []

    async def claim_cancellation(
        self,
        *,
        event_id: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        thread_ts: str,
        event_ts: str,
        streaming_message_timestamps: tuple[str, ...] = (),
    ) -> CancellationClaim:
        del team_id, channel_id, user_id, thread_ts, event_ts
        self.claimed_events.append(event_id)
        self.streaming_message_timestamps = streaming_message_timestamps
        return self.claim


async def test_dispatcher_serializes_conversation_key_and_deterministic_id() -> None:
    client = FakeInngest()
    dispatcher = InngestQuestionDispatcher(client)  # type: ignore[arg-type]
    job = QuestionJob(
        event_id="Ev1",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="2.0",
        thread_ts="1.0",
        question="What changed?",
    )

    result = await dispatcher.enqueue(job)

    assert result == ["event-result"]
    assert client.event is not None
    event: Any = client.event
    assert event.id == "slack-question:Ev1"
    assert event.data["conversation_id"] == "T1:C1:1.0"


async def test_follow_up_dispatcher_uses_stable_event_identity() -> None:
    client = FakeInngest()
    dispatcher = InngestFollowUpCandidateDispatcher(client)  # type: ignore[arg-type]
    candidate = FollowUpCandidateJob(
        event_id="Ev2",
        team_id="T1",
        channel_id="C1",
        user_id="U2",
        message_ts="3.0",
        thread_ts="1.0",
        message_text="Why?",
    )

    await dispatcher.enqueue_candidate(candidate)

    assert client.event is not None
    event: Any = client.event
    assert event.id == "slack-follow-up:Ev2"
    assert event.name == FOLLOW_UP_CANDIDATE_EVENT
    assert event.data["conversation_id"] == "T1:C1:1.0"


async def test_stop_handoff_persists_request_before_emitting_cancel_event() -> None:
    run_id = uuid4()
    client = FakeInngest()
    ledger = FakeCancellationLedger(CancellationClaim(run_id=run_id, accepted=True))
    handoff = InngestAgentSessionStopHandoff(
        client,  # type: ignore[arg-type]
        ledger=ledger,  # type: ignore[arg-type]
    )
    request = AgentSessionStopRequest(
        event_id="EvStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="4.0",
        streaming_message_ts=("2.0",),
    )

    await handoff.enqueue_stop(request)

    assert ledger.claimed_events == ["EvStop"]
    assert ledger.streaming_message_timestamps == ("2.0",)
    assert client.event is not None
    event: Any = client.event
    assert event.id == "slack-stop:EvStop"
    assert event.name == QUESTION_CANCELLED_EVENT
    assert event.data["agent_run_id"] == str(run_id)
    assert event.data["cancellation_accepted"] is True


async def test_stop_handoff_emits_persisted_too_late_outcome() -> None:
    run_id: UUID = uuid4()
    client = FakeInngest()
    ledger = FakeCancellationLedger(CancellationClaim(run_id=run_id, accepted=False))
    handoff = InngestAgentSessionStopHandoff(
        client,  # type: ignore[arg-type]
        ledger=ledger,  # type: ignore[arg-type]
    )
    request = AgentSessionStopRequest(
        event_id="EvLateStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="5.0",
        streaming_message_ts=("2.0",),
    )

    await handoff.enqueue_stop(request)

    assert client.event is not None
    event: Any = client.event
    assert event.data["agent_run_id"] == str(run_id)
    assert event.data["cancellation_accepted"] is False
