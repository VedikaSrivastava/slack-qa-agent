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
    expected_show_sources: bool | None = None
    forbidden_phrases: list[str] = Field(default_factory=list)
    max_tool_calls: int = Field(default=9, ge=0)
    max_model_calls: int | None = Field(default=None, ge=0)
    max_retrieval_rounds: int = Field(default=2, ge=0, le=2)
    insufficient_evidence_acceptable: bool = False


class CheckResult(BaseModel):
    name: str
    passed: bool
    details: str = ""


class HitCount(BaseModel):
    """How many of one gold-label group's fragments were found in the answer."""

    matched: int = Field(ge=0)
    total: int = Field(ge=0)


class EvalResult(BaseModel):
    case_id: str
    passed: bool
    checks: list[CheckResult]
    answer: str
    # `source_ids` are citations in the final answer; retrieved IDs are kept separate so
    # retrieval failures can be distinguished from citation failures.
    source_ids: list[str]
    retrieved_artifact_ids: list[str]
    show_sources: bool = False
    tool_call_count: int
    retrieval_round_count: int
    model_call_count: int
    input_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    duration_ms: int | None = Field(default=None, ge=0)
    # Response-length signal for the "is the agent too terse / too verbose" axis. Measured on
    # the delivered answer text with citation markers still present.
    answer_chars: int = Field(default=0, ge=0)
    answer_words: int = Field(default=0, ge=0)
    # Fragment-level gold-label hits per group (facts/entities/dates/commands/customers). The
    # binary `checks` say "did every fragment match"; this says "how many". A continuous
    # accuracy proxy that does not collapse to 0 when one of seven fact fragments is
    # paraphrased.
    deterministic_hits: dict[str, HitCount] = Field(default_factory=dict)
