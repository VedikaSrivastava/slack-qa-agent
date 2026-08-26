"""Stable application boundary shared by Slack, Inngest, CLI, and evals."""

from __future__ import annotations

from typing import Protocol

from knowledge_assistant.agent.models import AgentResponse


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
