"""Bounded conversational state persisted by LangGraph."""

from __future__ import annotations

from typing import Any, TypedDict


class ConversationTurn(TypedDict):
    question: str
    answer: str


class AgentState(TypedDict, total=False):
    question: str
    standalone_question: str
    agent_run_id: str
    conversation_id: str
    search_queries: list[str]
    account_lookup: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    history: list[ConversationTurn]
    retrieval_round_count: int
    tool_call_count: int
    input_tokens: int
    output_tokens: int
    evidence_sufficient: bool
    insufficiency_reason: str
    draft_answer: str
    final_answer: str
    grounding_valid: bool
    grounding_issues: list[str]
    repair_attempted: bool
