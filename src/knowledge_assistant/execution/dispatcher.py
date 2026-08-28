"""Execution dispatch interfaces and Inngest implementation."""

from __future__ import annotations

from typing import Protocol

import inngest

from knowledge_assistant.execution.models import (
    AgentSessionStopRequest,
    FollowUpCandidateJob,
    QuestionCancellationJob,
    QuestionJob,
)
from knowledge_assistant.persistence.repositories import RunLedger

QUESTION_RECEIVED_EVENT = "slack/question.received"
FOLLOW_UP_CANDIDATE_EVENT = "slack/follow_up.candidate"
QUESTION_READY_EVENT = "slack/question.ready"
RESPONDER_CLASSIFICATION_EVENT = "slack/follow_up.classify"
QUESTION_CANCELLED_EVENT = "slack/question.cancelled"


class QuestionDispatcher(Protocol):
    async def enqueue(self, job: QuestionJob) -> list[str]:
        """Hand a job to durable execution and return the accepted event IDs."""
        ...


class InngestQuestionDispatcher:
    """Durable queue adapter.

    The event payload is sent as-is into Inngest and keyed by ``slack-question:<event_id>`` so
    Slack retry deliveries map to the same durable execution path.
    """

    def __init__(self, client: inngest.Inngest) -> None:
        self._client = client

    async def enqueue(self, job: QuestionJob) -> list[str]:
        # Keep the idempotency key local to the transport boundary so retries cannot branch run ids.
        return await self._client.send(
            inngest.Event(
                id=f"slack-question:{job.event_id}",
                name=QUESTION_RECEIVED_EVENT,
                data=job.model_dump(mode="json"),
            )
        )


class InngestFollowUpCandidateDispatcher:
    """Durably defer thread ownership and semantic responder judgment.

    Candidate events never trigger work directly. They are normalized into a routing function
    that can keep thread-ownership and model inference separated from Slack ingress latency.
    """

    def __init__(self, client: inngest.Inngest) -> None:
        self._client = client

    async def enqueue_candidate(self, job: FollowUpCandidateJob) -> list[str]:
        # Follow-up candidates use a different event namespace to avoid conflating idempotency
        # with explicit root mentions.
        return await self._client.send(
            inngest.Event(
                id=f"slack-follow-up:{job.event_id}",
                name=FOLLOW_UP_CANDIDATE_EVENT,
                data=job.model_dump(mode="json"),
            )
        )


class InngestAgentSessionStopHandoff:
    """Persist cooperative cancellation before emitting the matching Inngest event."""

    def __init__(self, client: inngest.Inngest, ledger: RunLedger) -> None:
        self._client = client
        self._ledger = ledger

    async def enqueue_stop(self, request: AgentSessionStopRequest) -> list[str]:
        # Persist the stop disposition first so cancellation cleanup can reason about it even if
        # the Inngest event fan-out is delayed or replayed.
        claim = await self._ledger.claim_cancellation(
            event_id=request.event_id,
            team_id=request.team_id,
            channel_id=request.channel_id,
            user_id=request.user_id,
            thread_ts=request.thread_ts,
            event_ts=request.event_ts,
            streaming_message_timestamps=request.streaming_message_ts,
        )
        job = QuestionCancellationJob(
            **request.model_dump(mode="python"),
            agent_run_id=claim.run_id,
            cancellation_accepted=claim.accepted,
        )
        return await self._client.send(
            inngest.Event(
                id=f"slack-stop:{request.event_id}",
                name=QUESTION_CANCELLED_EVENT,
                data=job.model_dump(mode="json"),
            )
        )
