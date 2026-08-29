"""Stable application boundary shared by Slack, Inngest, CLI, and evals."""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol

from knowledge_assistant.agent.models import AgentResponse, ProcessorEvent


class QuestionProcessor(Protocol):
    async def answer(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AgentResponse:
        """Answer a question without depending on a transport-specific request type."""
        ...


class StreamingQuestionProcessor(QuestionProcessor, Protocol):
    """Optional event-streaming extension that preserves the stable answer-only boundary."""

    def run(
        self,
        *,
        question: str,
        conversation_id: str,
        agent_run_id: str,
    ) -> AsyncIterator[ProcessorEvent]:
        """Yield sanitized progress followed by exactly one finalized response."""
        ...
