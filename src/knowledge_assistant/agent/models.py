"""Typed inputs, evidence, and outputs at the agent boundary."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceReference(BaseModel):
    artifact_id: str
    title: str
    score: float | None = None
    snippet: str | None = None


class AgentResponse(BaseModel):
    answer: str
    sources: list[EvidenceReference] = Field(default_factory=list)
    tool_call_count: int = Field(default=0, ge=0)
    retrieval_round_count: int = Field(default=0, ge=0, le=2)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    insufficient_evidence: bool = False
