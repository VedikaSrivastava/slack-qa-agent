"""LangGraph state shared by deterministic and agentic nodes."""

from __future__ import annotations

from typing import TypedDict


class EvidenceItem(TypedDict):
    artifact_id: str
    title: str
    snippet: str
    content: str
    score: float | None


class AgentState(TypedDict, total=False):
    question: str
    standalone_question: str
    search_queries: list[str]
    evidence: list[EvidenceItem]
    retrieval_attempts: int
    tool_call_count: int
    draft_answer: str
    final_answer: str
