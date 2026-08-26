"""Deterministic and model-backed nodes for the bounded LangGraph workflow."""

from __future__ import annotations

import json
from typing import Any, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from slack_qa_agent.agent.profiles import AgentProfile
from slack_qa_agent.agent.prompts import (
    GENERATE_ANSWER,
    GRADE_EVIDENCE,
    PLAN_RETRIEVAL,
    REPAIR_ANSWER,
    RESOLVE_QUESTION,
    SYSTEM_GROUNDING_RULES,
    VERIFY_GROUNDING,
)
from slack_qa_agent.agent.state import AgentState, ConversationTurn
from slack_qa_agent.agent.tools import KnowledgeTools
from slack_qa_agent.retrieval.models import (
    MAX_CONTEXT_CHARS,
    AccountLookupInput,
    ReadArtifactsInput,
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


class AgentNodes:
    def __init__(self, model: BaseChatModel, tools: KnowledgeTools, profile: AgentProfile) -> None:
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
        query_cap = (
            self._profile.max_initial_queries
            if state.get("retrieval_round_count", 0) == 0
            else self._profile.max_refined_queries
        )
        structured_evidence = []
        account_lookup = state.get("account_lookup")
        structured_lookup_allowed = (
            account_lookup is not None
            and state.get("retrieval_round_count", 0) == 0
            and remaining_tool_calls > 0
        )
        if structured_lookup_allowed:
            structured_evidence = await self._tools.lookup_accounts(
                AccountLookupInput.model_validate(account_lookup)
            )
            remaining_tool_calls -= 1

        # Reserve one tool call to read full artifacts whenever lexical search can run.
        query_limit = min(query_cap, max(0, remaining_tool_calls - 1))
        queries = state.get("search_queries", [])[:query_limit]
        hits = []
        for query in queries:
            hits.extend(
                await self._tools.search_knowledge(
                    SearchKnowledgeInput(query=query, limit=self._profile.search_limit)
                )
            )

        ranked_ids: list[str] = []
        for hit in sorted(hits, key=lambda item: item.score if item.score is not None else 0):
            if hit.artifact_id not in ranked_ids:
                ranked_ids.append(hit.artifact_id)
        ranked_ids = ranked_ids[: self._profile.max_artifacts]
        evidence = list(structured_evidence)
        read_calls = 0
        remaining_context = MAX_CONTEXT_CHARS - sum(len(item.content) for item in evidence)
        unread_ids = [
            artifact_id
            for artifact_id in ranked_ids
            if artifact_id not in {item.artifact_id for item in evidence}
        ]
        if unread_ids and remaining_context >= 1_000 and remaining_tool_calls > len(queries):
            evidence.extend(
                await self._tools.read_artifacts(
                    ReadArtifactsInput(
                        artifact_ids=unread_ids,
                        max_context_chars=remaining_context,
                    )
                )
            )
            read_calls = 1
        return {
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "retrieval_round_count": state.get("retrieval_round_count", 0) + 1,
            "tool_call_count": existing_tool_calls
            + len(queries)
            + read_calls
            + int(structured_lookup_allowed),
        }

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
