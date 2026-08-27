"""Bounded conversational state persisted by LangGraph."""

from __future__ import annotations

from typing import TypedDict

from knowledge_assistant.retrieval.models import JsonValue


class ConversationTurn(TypedDict):
    question: str
    answer: str


class AgentState(TypedDict, total=False):
    question: str
    standalone_question: str
    agent_run_id: str
    conversation_id: str
    search_queries: list[str]
    account_lookup: dict[str, JsonValue] | None
    evidence: list[dict[str, JsonValue]]
    history: list[ConversationTurn]
    retrieval_round_count: int
    tool_call_count: int
    model_call_count: int
    evidence_sufficient: bool
    insufficiency_reason: str
    draft_answer: str
    final_answer: str
    grounding_valid: bool
    grounding_issues: list[str]
