"""Typed inputs and outputs for the agent boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    conversation_id: str = Field(min_length=1, max_length=512)
    request_id: str = Field(min_length=1, max_length=512)


class Citation(BaseModel):
    artifact_id: str
    title: str
    snippet: str | None = None


class AgentResponse(BaseModel):
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    tool_call_count: int = 0
    latency_ms: int | None = None
