"""Typed inputs, evidence, and outputs at the agent boundary."""

from __future__ import annotations

from decimal import Decimal

from pydantic import BaseModel, Field


class AgentRequest(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)
    conversation_id: str = Field(min_length=1, max_length=512)
    agent_run_id: str = Field(min_length=1, max_length=512)


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
    estimated_cost_usd: Decimal | None = Field(default=None, ge=0)
    insufficient_evidence: bool = False

    @property
    def citations(self) -> list[EvidenceReference]:
        """Compatibility alias for callers from the initial scaffold."""

        return self.sources
