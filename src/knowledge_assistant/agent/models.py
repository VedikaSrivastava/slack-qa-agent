"""Typed inputs, evidence, and outputs at the agent boundary."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class EvidenceReference(BaseModel):
    artifact_id: str
    title: str
    score: float | None = None
    snippet: str | None = None


class QuestionDisposition(StrEnum):
    """User intent at the grounded-Q&A workflow boundary."""

    KNOWLEDGE_QUESTION = "knowledge_question"
    CAPABILITY_QUESTION = "capability_question"
    GREETING = "greeting"
    NEEDS_CLARIFICATION = "needs_clarification"
    OUT_OF_SCOPE = "out_of_scope"


class AgentResponse(BaseModel):
    answer: str
    disposition: QuestionDisposition = QuestionDisposition.KNOWLEDGE_QUESTION
    show_sources: bool = False
    sources: list[EvidenceReference] = Field(default_factory=list)
    retrieved_artifact_ids: list[str] = Field(default_factory=list)
    tool_call_count: int = Field(default=0, ge=0)
    model_call_count: int = Field(default=0, ge=0)
    retrieval_round_count: int = Field(default=0, ge=0, le=2)
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    insufficient_evidence: bool = False

    @property
    def requires_user_input(self) -> bool:
        """Whether delivery should leave the Slack agent session waiting on the user."""

        return self.disposition is QuestionDisposition.NEEDS_CLARIFICATION


class ProgressStage(StrEnum):
    """Small, code-owned workflow stages that are safe to show to users."""

    THINKING = "thinking"
    SEARCHING = "searching"
    REVIEWING = "reviewing"
    DRAFTING = "drafting"
    VERIFYING = "verifying"
    TIGHTENING = "tightening"


class ProgressEvent(BaseModel):
    """Sanitized progress signal; model text and graph payloads never cross this boundary."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["progress"] = "progress"
    agent_run_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    sequence: int = Field(ge=0)
    stage: ProgressStage
    retrieval_round: int | None = Field(default=None, ge=1, le=2)


class FinalAnswerEvent(BaseModel):
    """Terminal processor event emitted only after the graph finalizes a verified outcome."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["final_answer"] = "final_answer"
    agent_run_id: str = Field(min_length=1)
    event_id: str = Field(min_length=1)
    response: AgentResponse


type ProcessorEvent = ProgressEvent | FinalAnswerEvent
