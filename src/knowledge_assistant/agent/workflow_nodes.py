"""Deterministic and model-backed nodes for the grounded answer workflow."""

from __future__ import annotations

import json
from typing import TypedDict, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field, field_validator, model_validator

from knowledge_assistant.agent.citations import citation_issues
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
    MAX_ARTIFACT_BATCH,
    MAX_CONTEXT_CHARS,
    AccountLookupInput,
    EvidenceItem,
    JsonValue,
    ReadArtifactsInput,
    SearchHit,
    SearchKnowledgeInput,
)

MAX_INITIAL_QUERIES = 3
MAX_REFINED_QUERIES = 2
MAX_EVIDENCE_PAYLOAD_CHARS = 32_000
INSUFFICIENT_EVIDENCE_ANSWER = "I couldn't answer this from the knowledge base."


class ModelCallBudgetExceededError(RuntimeError):
    """Raised before a workflow can exceed its code-reviewed model-call budget."""


class PromptEvidence(TypedDict):
    artifact_id: str
    title: str
    content: str


class StandaloneQuestion(BaseModel):
    question: str = Field(min_length=1, max_length=8_000)

    @field_validator("question")
    @classmethod
    def require_question_text(cls, question: str) -> str:
        normalized = " ".join(question.split())
        if not normalized:
            raise ValueError("standalone question must contain text")
        return normalized


class RetrievalPlan(BaseModel):
    queries: list[str] = Field(min_length=1, max_length=MAX_INITIAL_QUERIES)
    account_lookup: AccountLookupInput | None = None

    @field_validator("queries")
    @classmethod
    def require_searchable_queries(cls, queries: list[str]) -> list[str]:
        normalized = [" ".join(query.split()) for query in queries]
        if any(not query for query in normalized):
            raise ValueError("retrieval queries must contain searchable text")
        return list(dict.fromkeys(normalized))


class EvidenceGrade(BaseModel):
    sufficient: bool
    reason: str = Field(min_length=1, max_length=1_000)
    refined_queries: list[str] = Field(default_factory=list, max_length=MAX_REFINED_QUERIES)

    @field_validator("refined_queries")
    @classmethod
    def normalize_refined_queries(cls, queries: list[str]) -> list[str]:
        normalized = [" ".join(query.split()) for query in queries]
        if any(not query for query in normalized):
            raise ValueError("refined queries must contain searchable text")
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def require_queries_when_insufficient(self) -> EvidenceGrade:
        if not self.sufficient and not self.refined_queries:
            raise ValueError("insufficient evidence requires at least one refined query")
        return self


class GroundingVerdict(BaseModel):
    valid: bool
    issues: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def require_issues_when_invalid(self) -> GroundingVerdict:
        if not self.valid and not self.issues:
            raise ValueError("invalid grounding verdict requires at least one issue")
        return self


def _evidence_payload(state: AgentState) -> str:
    """Serialize only prompt-required evidence within an escaped JSON character budget."""

    prompt_evidence: list[PromptEvidence] = []
    for raw_item in state.get("evidence", []):
        item = EvidenceItem.model_validate(raw_item)
        candidate = PromptEvidence(
            artifact_id=item.artifact_id,
            title=item.title,
            content=item.content,
        )
        candidate_payload = json.dumps(
            [*prompt_evidence, candidate], ensure_ascii=False, separators=(",", ":")
        )
        if len(candidate_payload) <= MAX_EVIDENCE_PAYLOAD_CHARS:
            prompt_evidence.append(candidate)
            continue

        # JSON escaping can expand quotes and backslashes, so calculate the largest safe prefix
        # against the serialized payload instead of subtracting raw string lengths.
        lower_bound = 0
        upper_bound = len(item.content)
        bounded_candidate: PromptEvidence | None = None
        while lower_bound <= upper_bound:
            midpoint = (lower_bound + upper_bound) // 2
            truncated_candidate = PromptEvidence(
                artifact_id=item.artifact_id,
                title=item.title,
                content=item.content[:midpoint],
            )
            truncated_payload = json.dumps(
                [*prompt_evidence, truncated_candidate],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            if len(truncated_payload) <= MAX_EVIDENCE_PAYLOAD_CHARS:
                bounded_candidate = truncated_candidate
                lower_bound = midpoint + 1
            else:
                upper_bound = midpoint - 1
        if bounded_candidate is not None:
            prompt_evidence.append(bounded_candidate)
        break

    return json.dumps(prompt_evidence, ensure_ascii=False, separators=(",", ":"))


def _rank_unique_artifact_ids(hits: list[SearchHit], limit: int) -> list[str]:
    """Keep SQLite BM25 order, where lower (often more negative) scores rank first."""

    ranked_ids: list[str] = []
    for hit in sorted(hits, key=lambda item: item.score if item.score is not None else 0):
        if hit.artifact_id not in ranked_ids:
            ranked_ids.append(hit.artifact_id)
    return ranked_ids[:limit]


def _merge_evidence(
    existing_evidence: list[EvidenceItem],
    new_evidence: list[EvidenceItem],
    *,
    max_artifacts: int,
) -> list[EvidenceItem]:
    """Merge evidence in discovery order while enforcing global workflow budgets."""

    merged_evidence: list[EvidenceItem] = []
    seen_artifact_ids: set[str] = set()
    remaining_context_chars = MAX_CONTEXT_CHARS
    for item in (*existing_evidence, *new_evidence):
        if item.artifact_id in seen_artifact_ids:
            continue
        if len(merged_evidence) >= max_artifacts or remaining_context_chars <= 0:
            break

        bounded_item = item
        if len(item.content) > remaining_context_chars:
            bounded_item = item.model_copy(
                update={"content": item.content[:remaining_context_chars]}
            )
        merged_evidence.append(bounded_item)
        seen_artifact_ids.add(item.artifact_id)
        remaining_context_chars -= len(bounded_item.content)
    return merged_evidence


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

    def _next_model_call_count(self, state: AgentState) -> int:
        model_call_count = state.get("model_call_count", 0)
        if model_call_count >= self._profile.max_model_calls:
            raise ModelCallBudgetExceededError(
                f"Agent profile {self._profile.name!r} exhausted its model-call budget"
            )
        return model_call_count + 1

    async def resolve_question(self, state: AgentState) -> AgentState:
        history = state.get("history", [])[-self._profile.max_history_turns :]
        if not history:
            return {"standalone_question": state["question"], "history": []}
        model_call_count = self._next_model_call_count(state)
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
        return {
            "standalone_question": parsed.question,
            "history": history,
            "model_call_count": model_call_count,
        }

    async def plan_retrieval(self, state: AgentState) -> AgentState:
        model_call_count = self._next_model_call_count(state)
        planner = self._model.with_structured_output(RetrievalPlan)
        result = await planner.ainvoke(
            [
                SystemMessage(content=PLAN_RETRIEVAL),
                HumanMessage(content=state["standalone_question"]),
            ]
        )
        parsed = cast(RetrievalPlan, result)
        return {
            "search_queries": parsed.queries[: self._profile.max_initial_queries],
            "account_lookup": cast(
                dict[str, JsonValue], parsed.account_lookup.model_dump(mode="json")
            )
            if parsed.account_lookup
            else None,
            "model_call_count": model_call_count,
        }

    async def execute_retrieval(self, state: AgentState) -> AgentState:
        existing_tool_calls = state.get("tool_call_count", 0)
        remaining_tool_calls = max(0, self._profile.max_tool_calls - existing_tool_calls)
        account_evidence, account_lookup_calls = await self._lookup_account_evidence(
            state,
            remaining_tool_calls,
        )
        remaining_tool_calls -= account_lookup_calls
        previous_evidence = [
            EvidenceItem.model_validate(item) for item in state.get("evidence", [])
        ]
        evidence_before_lexical_read = _merge_evidence(
            previous_evidence,
            account_evidence,
            max_artifacts=self._profile.max_artifacts,
        )

        search_query_cap = (
            self._profile.max_initial_queries
            if state.get("retrieval_round_count", 0) == 0
            else self._profile.max_refined_queries
        )
        # Reserve one tool call to read full artifacts whenever lexical search can run.
        can_add_lexical_evidence = (
            len(evidence_before_lexical_read) < self._profile.max_artifacts
            and sum(len(item.content) for item in evidence_before_lexical_read)
            <= MAX_CONTEXT_CHARS - 1_000
        )
        search_query_limit = (
            min(search_query_cap, max(0, remaining_tool_calls - 1))
            if can_add_lexical_evidence
            else 0
        )
        search_queries = state.get("search_queries", [])[:search_query_limit]
        search_hits = await self._search_knowledge(search_queries)
        ranked_artifact_ids = _rank_unique_artifact_ids(
            search_hits,
            self._profile.max_artifacts,
        )
        evidence, artifact_read_calls = await self._read_unseen_artifacts(
            evidence_before_lexical_read,
            ranked_artifact_ids,
            can_read=remaining_tool_calls > len(search_queries),
        )
        return {
            "evidence": [
                cast(dict[str, JsonValue], item.model_dump(mode="json")) for item in evidence
            ],
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
        request = AccountLookupInput.model_validate(account_lookup)
        bounded_request = request.model_copy(
            update={"limit": min(request.limit, self._profile.max_artifacts)}
        )
        evidence = await self._tools.lookup_accounts(bounded_request)
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
        ][: min(MAX_ARTIFACT_BATCH, max(0, self._profile.max_artifacts - len(evidence)))]
        remaining_context_chars = MAX_CONTEXT_CHARS - sum(len(item.content) for item in evidence)
        if not unread_artifact_ids or remaining_context_chars < 1_000 or not can_read:
            return evidence, 0
        newly_read_evidence = await self._tools.read_artifacts(
            ReadArtifactsInput(
                artifact_ids=unread_artifact_ids,
                max_context_chars=remaining_context_chars,
            )
        )
        return (
            _merge_evidence(
                evidence,
                newly_read_evidence,
                max_artifacts=self._profile.max_artifacts,
            ),
            1,
        )

    async def grade_evidence(self, state: AgentState) -> AgentState:
        model_call_count = self._next_model_call_count(state)
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
            "model_call_count": model_call_count,
        }

    def route_after_grade(self, state: AgentState) -> str:
        if (
            state.get("evidence_sufficient")
            or state.get("retrieval_round_count", 0) >= self._profile.max_retrieval_rounds
        ):
            return "generate"
        return "refine"

    async def refine_retrieval(self, state: AgentState) -> AgentState:
        queries = [query for query in state.get("search_queries", []) if query.strip()]
        if not queries:
            queries = [state["standalone_question"]]
        return {"search_queries": queries[: self._profile.max_refined_queries]}

    async def generate_answer(self, state: AgentState) -> AgentState:
        if not state.get("evidence"):
            return {
                "draft_answer": INSUFFICIENT_EVIDENCE_ANSWER,
                "final_answer": INSUFFICIENT_EVIDENCE_ANSWER,
                "grounding_valid": True,
            }
        model_call_count = self._next_model_call_count(state)
        response = await self._model.ainvoke(
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{GENERATE_ANSWER}"),
                HumanMessage(
                    content=f"Question:\n{state['standalone_question']}\n\nEvidence:\n{_evidence_payload(state)}"
                ),
            ]
        )
        answer = str(response.content)
        # `final_answer` is reserved for terminal outcomes. A later repair must be able to replace
        # this draft without finalization preferring the rejected text.
        return {"draft_answer": answer, "model_call_count": model_call_count}

    async def verify_grounding(self, state: AgentState) -> AgentState:
        if not state.get("evidence"):
            return {"grounding_valid": True, "grounding_issues": []}
        evidence = [EvidenceItem.model_validate(item) for item in state["evidence"]]
        deterministic_issues = citation_issues(state["draft_answer"], evidence)
        if deterministic_issues:
            return {"grounding_valid": False, "grounding_issues": deterministic_issues}

        model_call_count = self._next_model_call_count(state)
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
        return {
            "grounding_valid": parsed.valid,
            "grounding_issues": parsed.issues,
            "model_call_count": model_call_count,
        }

    def route_after_verify(self, state: AgentState) -> str:
        return "finalize" if state.get("grounding_valid") else "repair"

    async def repair_answer(self, state: AgentState) -> AgentState:
        model_call_count = self._next_model_call_count(state)
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
        return {"draft_answer": str(response.content), "model_call_count": model_call_count}

    async def reject_ungrounded_answer(self, state: AgentState) -> AgentState:
        del state
        return {
            "final_answer": (
                "I couldn't produce an answer that was fully supported by the knowledge base."
            )
        }

    async def finalize(self, state: AgentState) -> AgentState:
        answer = state.get("final_answer") or state["draft_answer"]
        turn = ConversationTurn(question=state["question"], answer=answer)
        history = [*state.get("history", []), turn][-self._profile.max_history_turns :]
        return {"final_answer": answer, "history": history}
