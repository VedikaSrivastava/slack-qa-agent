"""Execution dispatch interfaces and Inngest implementation."""

from __future__ import annotations

from typing import Protocol

import inngest

from slack_qa_agent.execution.models import QuestionJob

QUESTION_RECEIVED_EVENT = "slack/question.received"


class QuestionDispatcher(Protocol):
    async def enqueue(self, job: QuestionJob) -> list[str]:
        """Persist a job for asynchronous processing and return execution event IDs."""
        ...


class InngestQuestionDispatcher:
    """Durable queue adapter. The Slack event ID is also the idempotency key."""

    def __init__(self, client: inngest.Inngest) -> None:
        self._client = client

    async def enqueue(self, job: QuestionJob) -> list[str]:
        return await self._client.send(
            inngest.Event(
                id=f"slack-question:{job.event_id}",
                name=QUESTION_RECEIVED_EVENT,
                data=job.model_dump(mode="json"),
            )
        )
