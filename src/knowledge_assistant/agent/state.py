"""Bounded conversational state persisted by LangGraph."""

from __future__ import annotations

from typing import Literal, NotRequired, TypedDict

from knowledge_assistant.agent.models import QuestionDisposition
from knowledge_assistant.retrieval.models import JsonValue


class ConversationTurn(TypedDict):
    agent_run_id: str
    question: str
    answer: str
    sources: NotRequired[list[dict[str, JsonValue]]]
    retrieved_artifact_ids: NotRequired[list[str]]


class AgentState(TypedDict, total=False):
    question: str
    standalone_question: str
    agent_run_id: str
    conversation_id: str
    question_disposition: QuestionDisposition
    show_sources: bool
    comparison_query: str | None
    search_queries: list[str]
    account_lookup: dict[str, JsonValue] | None
    account_lookup_coverage: dict[str, JsonValue] | None
    evidence: list[dict[str, JsonValue]]
    response_sources: list[dict[str, JsonValue]]
    response_mode: Literal["answer", "sources_only"]
    reuse_turn_id: str | None
    history: list[ConversationTurn]
    retrieval_round_count: int
    tool_call_count: int
    model_call_count: int
    input_tokens: int
    output_tokens: int
    evidence_sufficient: bool
    insufficiency_reason: str
    supported_question_parts: list[str]
    missing_question_parts: list[str]
    draft_answer: str
    final_answer: str
    # Code-owned terminal outcome. Internal retrieval/grounding graders are diagnostics and must
    # not cause a substantive user-facing answer to be mislabeled as an abstention.
    is_abstention: bool
    grounding_valid: bool
    grounding_issues: list[str]
