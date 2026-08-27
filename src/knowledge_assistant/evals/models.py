"""Evaluation case and result schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvalCase(BaseModel):
    id: str = Field(min_length=1, max_length=128)
    category: str = Field(min_length=1, max_length=128)
    prior_turns: list[str] = Field(default_factory=list, max_length=6)
    question: str = Field(min_length=1, max_length=8_000)
    reference_answer: str = Field(min_length=1, max_length=12_000)
    expected_facts: list[str] = Field(default_factory=list)
    expected_entities: list[str] = Field(default_factory=list)
    expected_dates: list[str] = Field(default_factory=list)
    expected_commands: list[str] = Field(default_factory=list)
    expected_customers: list[str] = Field(default_factory=list)
    expected_source_ids: list[str] = Field(default_factory=list)
    forbidden_phrases: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(default=9, ge=0)
    max_retrieval_rounds: int = Field(default=2, ge=0, le=2)
    insufficient_evidence_acceptable: bool = False


class CheckResult(BaseModel):
    name: str
    passed: bool
    details: str = ""


class EvalResult(BaseModel):
    case_id: str
    passed: bool
    checks: list[CheckResult]
    answer: str
    # `source_ids` are citations in the final answer; retrieved IDs are kept separate so
    # retrieval failures can be distinguished from citation failures.
    source_ids: list[str]
    retrieved_artifact_ids: list[str]
    tool_call_count: int
    retrieval_round_count: int
    model_call_count: int
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
