"""Deterministic and model-backed nodes for the grounded answer workflow."""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from knowledge_assistant.agent.profiles import AgentProfile
from knowledge_assistant.agent.prompts import (
    GENERATE_ANSWER,
    GRADE_EVIDENCE,
    PLAN_RETRIEVAL,
    REPAIR_ANSWER,
    RESOLVE_QUESTION,
    SYSTEM_GROUNDING_RULES,
    VERIFY_GROUNDING,
)
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.state import AgentState, ConversationTurn
from knowledge_assistant.retrieval.models import (
    MAX_CONTEXT_CHARS,
    AccountLookupInput,
    EvidenceItem,
    ReadArtifactsInput,
    SearchHit,
    SearchKnowledgeInput,
)

MAX_INITIAL_QUERIES = 3
MAX_REFINED_QUERIES = 2


class StandaloneQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)


class RetrievalPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=MAX_INITIAL_QUERIES)
    account_lookup: AccountLookupInput | None = None


class EvidenceGrade(BaseModel):
    sufficient: bool
    reason: str = Field(max_length=1_000)
    refined_queries: list[str] = Field(default_factory=list, max_length=MAX_REFINED_QUERIES)


class GroundingVerdict(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list, max_length=8)


def _evidence_payload(state: AgentState) -> str:
    return json.dumps(state.get("evidence", []), ensure_ascii=False, separators=(",", ":"))


def _rank_unique_artifact_ids(hits: list[SearchHit], limit: int) -> list[str]:
    ranked_ids: list[str] = []
    for hit in sorted(hits, key=lambda item: item.score if item.score is not None else 0):
        if hit.artifact_id not in ranked_ids:
            ranked_ids.append(hit.artifact_id)
    return ranked_ids[:limit]


class GroundedAnswerNodes:
    """Node implementations for one bounded, evidence-grounded answer workflow."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: KnowledgeRetrievalTools,
        profile: AgentProfile,
    ) -> None:
        self._model = model
        self._tools = tools
        self._profile = profile

    async def resolve_question(self, state: AgentState) -> dict[str, Any]:
        history = state.get("history", [])[-self._profile.max_history_turns :]
        if not history:
            return {"standalone_question": state["question"], "history": []}
        resolver = self._model.with_structured_output(StandaloneQuestion)
        result = await resolver.ainvoke(
            [
                SystemMessage(content=RESOLVE_QUESTION),
                HumanMessage(
                    content=json.dumps(
                        {"recent_turns": history, "current_message": state["question"]},
                        ensure_ascii=False,
                    )
                ),
            ]
        )
        parsed = cast(StandaloneQuestion, result)
        return {"standalone_question": parsed.question, "history": history}

    async def plan_retrieval(self, state: AgentState) -> dict[str, Any]:
        planner = self._model.with_structured_output(RetrievalPlan)
        result = await planner.ainvoke(
            [
                SystemMessage(content=PLAN_RETRIEVAL),
                HumanMessage(content=state["standalone_question"]),
            ]
        )
        parsed = cast(RetrievalPlan, result)
        queries = [" ".join(query.split()) for query in parsed.queries if query.strip()]
        return {
            "search_queries": list(dict.fromkeys(queries))[: self._profile.max_initial_queries],
            "account_lookup": parsed.account_lookup.model_dump(mode="json")
            if parsed.account_lookup
            else None,
        }

    async def execute_retrieval(self, state: AgentState) -> dict[str, Any]:
        existing_tool_calls = state.get("tool_call_count", 0)
        remaining_tool_calls = max(0, self._profile.max_tool_calls - existing_tool_calls)
        account_evidence, account_lookup_calls = await self._lookup_account_evidence(
            state,
            remaining_tool_calls,
        )
        remaining_tool_calls -= account_lookup_calls

        search_query_cap = (
            self._profile.max_initial_queries
            if state.get("retrieval_round_count", 0) == 0
            else self._profile.max_refined_queries
        )
        # Reserve one tool call to read full artifacts whenever lexical search can run.
        search_query_limit = min(search_query_cap, max(0, remaining_tool_calls - 1))
        search_queries = state.get("search_queries", [])[:search_query_limit]
        search_hits = await self._search_knowledge(search_queries)
        ranked_artifact_ids = _rank_unique_artifact_ids(
            search_hits,
            self._profile.max_artifacts,
        )
        evidence, artifact_read_calls = await self._read_unseen_artifacts(
            account_evidence,
            ranked_artifact_ids,
            can_read=remaining_tool_calls > len(search_queries),
        )
        return {
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "retrieval_round_count": state.get("retrieval_round_count", 0) + 1,
            "tool_call_count": existing_tool_calls
            + len(search_queries)
            + artifact_read_calls
            + account_lookup_calls,
        }

    async def _lookup_account_evidence(
        self,
        state: AgentState,
        remaining_tool_calls: int,
    ) -> tuple[list[EvidenceItem], int]:
        account_lookup = state.get("account_lookup")
        lookup_allowed = (
            account_lookup is not None
            and state.get("retrieval_round_count", 0) == 0
            and remaining_tool_calls > 0
        )
        if not lookup_allowed:
            return [], 0
        evidence = await self._tools.lookup_accounts(
            AccountLookupInput.model_validate(account_lookup)
        )
        return evidence, 1

    async def _search_knowledge(self, queries: list[str]) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for query in queries:
            hits.extend(
                await self._tools.search_knowledge(
                    SearchKnowledgeInput(query=query, limit=self._profile.search_limit)
                )
            )
        return hits

    async def _read_unseen_artifacts(
        self,
        existing_evidence: list[EvidenceItem],
        ranked_artifact_ids: list[str],
        *,
        can_read: bool,
    ) -> tuple[list[EvidenceItem], int]:
        evidence = list(existing_evidence)
        existing_artifact_ids = {item.artifact_id for item in evidence}
        unread_artifact_ids = [
            artifact_id
            for artifact_id in ranked_artifact_ids
            if artifact_id not in existing_artifact_ids
        ]
        remaining_context_chars = MAX_CONTEXT_CHARS - sum(len(item.content) for item in evidence)
        if not unread_artifact_ids or remaining_context_chars < 1_000 or not can_read:
            return evidence, 0
        evidence.extend(
            await self._tools.read_artifacts(
                ReadArtifactsInput(
                    artifact_ids=unread_artifact_ids,
                    max_context_chars=remaining_context_chars,
                )
            )
        )
        return evidence, 1

    async def grade_evidence(self, state: AgentState) -> dict[str, Any]:
        if not state.get("evidence"):
            return {
                "evidence_sufficient": False,
                "insufficiency_reason": "No relevant artifacts were retrieved.",
            }
        grader = self._model.with_structured_output(EvidenceGrade)
        result = await grader.ainvoke(
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{GRADE_EVIDENCE}"),
                HumanMessage(
                    content=f"Question:\n{state['standalone_question']}\n\nEvidence:\n{_evidence_payload(state)}"
                ),
            ]
        )
        parsed = cast(EvidenceGrade, result)
        return {
            "evidence_sufficient": parsed.sufficient,
            "insufficiency_reason": parsed.reason,
            "search_queries": parsed.refined_queries[:MAX_REFINED_QUERIES],
        }

    def route_after_grade(self, state: AgentState) -> str:
        if (
            state.get("evidence_sufficient")
            or state.get("retrieval_round_count", 0) >= self._profile.max_retrieval_rounds
        ):
            return "generate"
        return "refine"

    async def refine_retrieval(self, state: AgentState) -> dict[str, Any]:
        queries = [query for query in state.get("search_queries", []) if query.strip()]
        if not queries:
            queries = [state["standalone_question"]]
        return {"search_queries": queries[: self._profile.max_refined_queries]}

    async def generate_answer(self, state: AgentState) -> dict[str, Any]:
        if not state.get("evidence"):
            reason = state.get("insufficiency_reason", "No supporting evidence was found.")
            answer = f"I couldn't answer this from the knowledge base. {reason}"
            return {"draft_answer": answer, "final_answer": answer, "grounding_valid": True}
        response = await self._model.ainvoke(
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{GENERATE_ANSWER}"),
                HumanMessage(
                    content=f"Question:\n{state['standalone_question']}\n\nEvidence:\n{_evidence_payload(state)}"
                ),
            ]
        )
        answer = str(response.content)
        return {"draft_answer": answer, "final_answer": answer}

    async def verify_grounding(self, state: AgentState) -> dict[str, Any]:
        if not state.get("evidence"):
            return {"grounding_valid": True, "grounding_issues": []}
        verifier = self._model.with_structured_output(GroundingVerdict)
        result = await verifier.ainvoke(
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{VERIFY_GROUNDING}"),
                HumanMessage(
                    content=(
                        f"Question:\n{state['standalone_question']}\n\n"
                        f"Draft:\n{state['draft_answer']}\n\nEvidence:\n{_evidence_payload(state)}"
                    )
                ),
            ]
        )
        parsed = cast(GroundingVerdict, result)
        return {"grounding_valid": parsed.valid, "grounding_issues": parsed.issues}

    def route_after_verify(self, state: AgentState) -> str:
        return "finalize" if state.get("grounding_valid") else "repair"

    async def repair_answer(self, state: AgentState) -> dict[str, Any]:
        response = await self._model.ainvoke(
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{REPAIR_ANSWER}"),
                HumanMessage(
                    content=(
                        f"Question:\n{state['standalone_question']}\n\n"
                        f"Draft:\n{state['draft_answer']}\n\n"
                        f"Audit issues:\n{json.dumps(state.get('grounding_issues', []))}\n\n"
                        f"Evidence:\n{_evidence_payload(state)}"
                    )
                ),
            ]
        )
        return {"final_answer": str(response.content), "repair_attempted": True}

    async def finalize(self, state: AgentState) -> dict[str, Any]:
        answer = state.get("final_answer") or state["draft_answer"]
        turn = ConversationTurn(question=state["question"], answer=answer)
        history = [*state.get("history", []), turn][-self._profile.max_history_turns :]
        return {"final_answer": answer, "history": history}
