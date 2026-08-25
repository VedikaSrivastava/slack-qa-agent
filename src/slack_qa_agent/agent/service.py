"""Agent service boundary used by Slack, Inngest, CLI, and eval runners."""

from __future__ import annotations

from typing import Protocol

from slack_qa_agent.agent.models import AgentRequest, AgentResponse


class AgentService(Protocol):
    async def answer(self, request: AgentRequest) -> AgentResponse:
        """Answer one question using conversation state and grounded evidence."""
        ...
