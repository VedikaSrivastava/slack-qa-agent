"""Deterministic and model-backed nodes for the grounded answer workflow."""

from __future__ import annotations

import json
import re
from typing import Literal, TypedDict, TypeVar, cast

import structlog
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from knowledge_assistant.agent.citations import (
    citation_issues,
    hide_artifact_citations,
    references_for_cited_evidence,
)
from knowledge_assistant.agent.models import EvidenceReference, QuestionDisposition
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
GREETING_ANSWER = (
    "Hi! I can answer questions grounded in the internal knowledge base. "
    "What would you like to know?"
)
OUT_OF_SCOPE_ANSWER = (
    "I can help with questions answered from the internal knowledge base, but I can't handle "
    "that request. Ask me about a customer, product, incident, process, or other documented "
    "internal topic."
)
CAPABILITY_ANSWER = (
    "I answer read-only questions using the internal knowledge base. I can search documented "
    "customers, products, incidents, processes, and other internal topics, and I can show the "
    "supporting sources when you ask for them."
)
NO_SAVED_SOURCES_ANSWER = "I don't have saved sources for an earlier answer in this thread."
SAVED_SOURCES_ANSWER = "Here are the sources used for that earlier answer."

logger = structlog.get_logger(__name__)

_SIMPLE_GREETINGS = frozenset(
    {
        "good afternoon",
        "good evening",
        "good morning",
        "hello",
        "hello there",
        "hey",
        "hey there",
        "hi",
        "hi there",
        "hiya",
        "howdy",
    }
)


class ModelCallBudgetExceededError(RuntimeError):
    """Raised before a workflow can exceed its code-reviewed model-call budget."""


class StructuredOutputValidationError(ValueError):
    """Preserve raw usage metadata when a structured model response cannot be parsed."""

    def __init__(
        self,
        message: str,
        *,
        raw_response: object | None = None,
        model_call_count: int | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.model_call_count = model_call_count
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


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
    disposition: QuestionDisposition
    show_sources: bool = False
    response_mode: Literal["answer", "sources_only"] = "answer"
    queries: list[str] = Field(default_factory=list, max_length=MAX_INITIAL_QUERIES)
    account_lookup: AccountLookupInput | None = None
    clarification_question: str | None = Field(default=None, min_length=1, max_length=300)
    reuse_turn_id: str | None = Field(default=None, min_length=1, max_length=256)

    @field_validator("queries")
    @classmethod
    def require_searchable_queries(cls, queries: list[str]) -> list[str]:
        normalized = [" ".join(query.split()) for query in queries]
        if any(not query for query in normalized):
            raise ValueError("retrieval queries must contain searchable text")
        return list(dict.fromkeys(normalized))

    @field_validator("clarification_question")
    @classmethod
    def normalize_clarification_question(cls, question: str | None) -> str | None:
        if question is None:
            return None
        normalized = " ".join(question.split()).rstrip(".!?")
        if not normalized:
            raise ValueError("clarification question must contain text")
        return f"{normalized[:299]}?"

    @model_validator(mode="after")
    def require_fields_for_disposition(self) -> RetrievalPlan:
        if self.disposition is QuestionDisposition.KNOWLEDGE_QUESTION:
            if self.clarification_question is not None:
                raise ValueError("knowledge questions cannot include a clarification question")
            if self.response_mode == "sources_only":
                if self.queries or self.account_lookup is not None:
                    raise ValueError("source-only responses cannot trigger retrieval")
                return self
            if not self.queries and self.account_lookup is None and self.reuse_turn_id is None:
                raise ValueError(
                    "knowledge questions require a query, account filter, or reusable prior turn"
                )
            return self
        if self.disposition is QuestionDisposition.NEEDS_CLARIFICATION:
            if self.clarification_question is None:
                raise ValueError("unclear questions require one clarification question")
            if self.queries or self.account_lookup is not None:
                raise ValueError("unclear questions cannot trigger retrieval")
            return self
        if self.queries or self.account_lookup is not None:
            raise ValueError("non-knowledge messages cannot trigger retrieval")
        if self.clarification_question is not None:
            raise ValueError("only unclear questions can include a clarification question")
        if self.reuse_turn_id is not None:
            raise ValueError("only knowledge questions can select a prior turn")
        if self.response_mode != "answer":
            raise ValueError("only knowledge questions can change response mode")
        return self


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


StructuredOutput = TypeVar("StructuredOutput", bound=BaseModel)


def _is_simple_greeting(message: str) -> bool:
    """Recognize only whole-message greetings so substantive questions still reach the model."""

    normalized = " ".join(re.sub(r"[\W_]+", " ", message.casefold()).split())
    return normalized in _SIMPLE_GREETINGS


def _token_usage(response: object | None) -> tuple[int, int]:
    """Read normalized token counts from a LangChain message without trusting dynamic metadata."""

    usage = getattr(response, "usage_metadata", None)
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    return (
        input_tokens if isinstance(input_tokens, int) and input_tokens >= 0 else 0,
        output_tokens if isinstance(output_tokens, int) and output_tokens >= 0 else 0,
    )


def _usage_state_update(state: AgentState, response: object | None) -> AgentState:
    input_tokens, output_tokens = _token_usage(response)
    return {
        "input_tokens": state.get("input_tokens", 0) + input_tokens,
        "output_tokens": state.get("output_tokens", 0) + output_tokens,
    }


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


def _rank_unique_artifact_ids(hit_groups: list[list[SearchHit]], limit: int) -> list[str]:
    """Preserve each planned query's best evidence before filling deeper lexical results."""

    ranked_groups = [
        sorted(group, key=lambda item: item.score if item.score is not None else 0)
        for group in hit_groups
    ]
    ranked_ids: list[str] = []
    for result_rank in range(max((len(group) for group in ranked_groups), default=0)):
        for group in ranked_groups:
            if result_rank >= len(group):
                continue
            artifact_id = group[result_rank].artifact_id
            if artifact_id not in ranked_ids:
                ranked_ids.append(artifact_id)
            if len(ranked_ids) == limit:
                return ranked_ids
    return ranked_ids


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


def _matching_prior_turn(
    history: list[ConversationTurn],
    *,
    reuse_turn_id: str | None,
) -> ConversationTurn | None:
    if reuse_turn_id is None:
        return None
    return next(
        (turn for turn in history if turn["agent_run_id"] == reuse_turn_id),
        None,
    )


def _sources_for_prior_turn(
    history: list[ConversationTurn],
    *,
    reuse_turn_id: str | None,
) -> list[dict[str, JsonValue]]:
    turn = _matching_prior_turn(
        history,
        reuse_turn_id=reuse_turn_id,
    )
    if turn is None:
        return []
    return [
        cast(
            dict[str, JsonValue],
            EvidenceReference.model_validate(source).model_dump(mode="json"),
        )
        for source in turn.get("sources", [])
    ]


def _artifact_ids_for_prior_turn(
    history: list[ConversationTurn],
    *,
    reuse_turn_id: str | None,
) -> list[str]:
    turn = _matching_prior_turn(history, reuse_turn_id=reuse_turn_id)
    if turn is None:
        return []
    return list(dict.fromkeys(turn.get("retrieved_artifact_ids", [])))


def _response_sources_from_state(state: AgentState) -> list[dict[str, JsonValue]]:
    explicit_sources = state.get("response_sources", [])
    if explicit_sources:
        return [
            cast(
                dict[str, JsonValue],
                EvidenceReference.model_validate(source).model_dump(mode="json"),
            )
            for source in explicit_sources
        ]

    answer = state.get("final_answer") or state.get("draft_answer", "")
    evidence = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
    return [
        cast(dict[str, JsonValue], source.model_dump(mode="json"))
        for source in references_for_cited_evidence(answer, evidence)
    ]


class GroundedAnswerNodes:
    """Node implementations for one bounded, evidence-grounded answer workflow."""

    def __init__(
        self,
        model: BaseChatModel,
        tools: KnowledgeRetrievalTools,
        profile: AgentProfile,
        *,
        answer_model: BaseChatModel | None = None,
    ) -> None:
        # `model` runs the structured classification nodes (resolve / plan / grade / verify).
        # `answer_model` runs `generate_answer` / `repair_answer`; it defaults to `model` so a
        # single-model profile is unchanged.
        self._model = model
        self._answer_model = answer_model or model
        self._tools = tools
        self._profile = profile

    def _next_model_call_count(self, state: AgentState) -> int:
        model_call_count = state.get("model_call_count", 0)
        if model_call_count >= self._profile.max_model_calls:
            raise ModelCallBudgetExceededError(
                f"Agent profile {self._profile.name!r} exhausted its model-call budget"
            )
        return model_call_count + 1

    async def _invoke_structured(
        self,
        output_type: type[StructuredOutput],
        messages: list[BaseMessage],
    ) -> tuple[StructuredOutput, object | None]:
        """Return parsed output plus its raw message so usage survives graph checkpoints."""

        structured_model = self._model.with_structured_output(output_type, include_raw=True)
        result = await structured_model.ainvoke(messages)
        # Test doubles and some compatible model wrappers may return the parsed object directly.
        if isinstance(result, output_type):
            return result, None
        if not isinstance(result, dict):
            raise StructuredOutputValidationError(
                f"Structured model returned unexpected {type(result).__name__}"
            )
        parsed = result.get("parsed")
        if not isinstance(parsed, output_type):
            parsing_error = result.get("parsing_error")
            if isinstance(parsing_error, BaseException):
                raise StructuredOutputValidationError(
                    "Structured model output could not be parsed",
                    raw_response=result.get("raw"),
                ) from parsing_error
            raise StructuredOutputValidationError(
                "Structured model output did not contain the expected parsed value",
                raw_response=result.get("raw"),
            )
        return parsed, result.get("raw")

    async def _invoke_structured_with_retry(
        self,
        output_type: type[StructuredOutput],
        messages: list[BaseMessage],
        *,
        state: AgentState,
    ) -> tuple[StructuredOutput, AgentState]:
        """Re-ask once for an invalid schema, then surface a typed terminal failure.

        Every attempted call consumes the hard model-call budget and contributes any available
        token metadata. The second invalid response is never converted into guessed behavior.
        """

        model_call_count = self._next_model_call_count(state)
        input_tokens = state.get("input_tokens", 0)
        output_tokens = state.get("output_tokens", 0)
        try:
            parsed, raw_response = await self._invoke_structured(output_type, messages)
            call_input_tokens, call_output_tokens = _token_usage(raw_response)
            return parsed, {
                "model_call_count": model_call_count,
                "input_tokens": input_tokens + call_input_tokens,
                "output_tokens": output_tokens + call_output_tokens,
            }
        except (ValidationError, ValueError) as invalid:
            failed_input_tokens, failed_output_tokens = _token_usage(
                getattr(invalid, "raw_response", None)
            )
            input_tokens += failed_input_tokens
            output_tokens += failed_output_tokens
            logger.warning(
                "structured_output_retry_started",
                agent_run_id=state.get("agent_run_id"),
                output_schema=output_type.__name__,
                model_call_count=model_call_count,
                exception_class=type(invalid).__name__,
            )
            reason = " ".join(str(invalid).split())[:500]
            retry_messages = [
                *messages,
                HumanMessage(
                    content=(
                        f"That response was rejected by schema validation: {reason}. "
                        f"Return a valid {output_type.__name__} that obeys every rule."
                    )
                ),
            ]
            retry_state: AgentState = {**state, "model_call_count": model_call_count}
            model_call_count = self._next_model_call_count(retry_state)
            try:
                parsed, raw_response = await self._invoke_structured(output_type, retry_messages)
                call_input_tokens, call_output_tokens = _token_usage(raw_response)
                input_tokens += call_input_tokens
                output_tokens += call_output_tokens
                return parsed, {
                    "model_call_count": model_call_count,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                }
            except (ValidationError, ValueError) as retry_invalid:
                retry_input_tokens, retry_output_tokens = _token_usage(
                    getattr(retry_invalid, "raw_response", None)
                )
                input_tokens += retry_input_tokens
                output_tokens += retry_output_tokens
                logger.error(
                    "structured_output_validation_failed",
                    agent_run_id=state.get("agent_run_id"),
                    output_schema=output_type.__name__,
                    model_call_count=model_call_count,
                    exception_class=type(retry_invalid).__name__,
                    error_code="structured_output_invalid",
                )
                raise StructuredOutputValidationError(
                    f"{output_type.__name__} remained invalid after one repair attempt",
                    raw_response=getattr(retry_invalid, "raw_response", None),
                    model_call_count=model_call_count,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                ) from retry_invalid

    async def resolve_question(self, state: AgentState) -> AgentState:
        history = state.get("history", [])[-self._profile.max_history_turns :]
        if _is_simple_greeting(state["question"]):
            return {
                "standalone_question": state["question"],
                "history": history,
                "question_disposition": QuestionDisposition.GREETING,
                "final_answer": GREETING_ANSWER,
                "grounding_valid": True,
            }
        if not history:
            return {"standalone_question": state["question"], "history": []}
        model_call_count = self._next_model_call_count(state)
        parsed, raw_response = await self._invoke_structured(
            StandaloneQuestion,
            [
                SystemMessage(content=RESOLVE_QUESTION),
                HumanMessage(
                    content=json.dumps(
                        {
                            "recent_turns": [
                                {"question": turn["question"], "answer": turn["answer"]}
                                for turn in history
                            ],
                            "current_message": state["question"],
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
        )
        return {
            "standalone_question": parsed.question,
            "history": history,
            "model_call_count": model_call_count,
            **_usage_state_update(state, raw_response),
        }

    def route_after_resolution(self, state: AgentState) -> str:
        if QuestionDisposition(state["question_disposition"]) is QuestionDisposition.GREETING:
            return "finalize"
        return "plan"

    async def plan_retrieval(self, state: AgentState) -> AgentState:
        standalone_question = state["standalone_question"]
        prior_turns = [
            {
                "agent_run_id": turn["agent_run_id"],
                "question": turn["question"],
                "has_sources": bool(turn.get("sources")),
                "retrieved_artifact_count": len(turn.get("retrieved_artifact_ids", [])),
            }
            for turn in state.get("history", [])
        ]
        parsed, invocation_update = await self._invoke_structured_with_retry(
            RetrievalPlan,
            [
                SystemMessage(content=PLAN_RETRIEVAL),
                HumanMessage(
                    content=json.dumps(
                        {
                            "question": standalone_question,
                            "prior_turns": prior_turns,
                        },
                        ensure_ascii=False,
                    )
                ),
            ],
            state=state,
        )
        available_turn_ids = {turn["agent_run_id"] for turn in state.get("history", [])}
        if parsed.reuse_turn_id is not None and parsed.reuse_turn_id not in available_turn_ids:
            raise StructuredOutputValidationError(
                "Retrieval plan selected an unavailable prior turn",
                model_call_count=invocation_update.get("model_call_count"),
                input_tokens=invocation_update.get("input_tokens"),
                output_tokens=invocation_update.get("output_tokens"),
            )
        update: AgentState = {
            "question_disposition": parsed.disposition,
            "show_sources": parsed.show_sources,
            "response_mode": parsed.response_mode,
            "reuse_turn_id": parsed.reuse_turn_id,
            "search_queries": parsed.queries[: self._profile.max_initial_queries],
            "account_lookup": cast(
                dict[str, JsonValue], parsed.account_lookup.model_dump(mode="json")
            )
            if parsed.account_lookup
            else None,
            **invocation_update,
        }
        if parsed.disposition is QuestionDisposition.GREETING:
            update.update({"final_answer": GREETING_ANSWER, "grounding_valid": True})
        elif parsed.response_mode == "sources_only":
            response_sources = _sources_for_prior_turn(
                state.get("history", []),
                reuse_turn_id=parsed.reuse_turn_id,
            )
            update.update(
                {
                    "final_answer": (
                        SAVED_SOURCES_ANSWER if response_sources else NO_SAVED_SOURCES_ANSWER
                    ),
                    "show_sources": bool(response_sources),
                    "response_sources": response_sources,
                    "grounding_valid": True,
                }
            )
        elif parsed.disposition is QuestionDisposition.CAPABILITY_QUESTION:
            update.update({"final_answer": CAPABILITY_ANSWER, "grounding_valid": True})
        elif parsed.disposition is QuestionDisposition.NEEDS_CLARIFICATION:
            update.update(
                {
                    "final_answer": cast(str, parsed.clarification_question),
                    "grounding_valid": True,
                }
            )
        elif parsed.disposition is QuestionDisposition.OUT_OF_SCOPE:
            update.update({"final_answer": OUT_OF_SCOPE_ANSWER, "grounding_valid": True})
        return update

    def route_after_plan(self, state: AgentState) -> str:
        if state.get("response_mode") == "sources_only":
            return "finalize"
        if (
            QuestionDisposition(state["question_disposition"])
            is QuestionDisposition.KNOWLEDGE_QUESTION
        ):
            return "retrieve"
        return "finalize"

    async def execute_retrieval(self, state: AgentState) -> AgentState:
        existing_tool_calls = state.get("tool_call_count", 0)
        remaining_tool_calls = max(0, self._profile.max_tool_calls - existing_tool_calls)
        current_evidence = [EvidenceItem.model_validate(item) for item in state.get("evidence", [])]
        current_artifact_ids = {item.artifact_id for item in current_evidence}
        reused_artifact_ids = _artifact_ids_for_prior_turn(
            state.get("history", []),
            reuse_turn_id=state.get("reuse_turn_id"),
        )
        # A refinement round receives cumulative evidence. Only reread prior artifacts that were
        # not already restored in an earlier round, otherwise reuse consumes budget twice.
        reused_artifact_ids = [
            artifact_id
            for artifact_id in reused_artifact_ids
            if artifact_id not in current_artifact_ids
        ][:MAX_ARTIFACT_BATCH]
        reused_evidence: list[EvidenceItem] = []
        reuse_read_calls = 0
        if reused_artifact_ids and remaining_tool_calls > 0:
            reused_evidence = await self._tools.read_artifacts(
                ReadArtifactsInput(artifact_ids=reused_artifact_ids)
            )
            reuse_read_calls = 1
            remaining_tool_calls -= 1
        previous_evidence = _merge_evidence(
            current_evidence,
            reused_evidence,
            max_artifacts=self._profile.max_artifacts,
        )
        account_evidence, account_lookup_calls = await self._lookup_account_evidence(
            state,
            remaining_tool_calls,
        )
        remaining_tool_calls -= account_lookup_calls
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
        search_hit_groups = await self._search_knowledge(search_queries)
        ranked_artifact_ids = _rank_unique_artifact_ids(
            search_hit_groups,
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
            + reuse_read_calls
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

    async def _search_knowledge(self, queries: list[str]) -> list[list[SearchHit]]:
        hit_groups: list[list[SearchHit]] = []
        for query in queries:
            hit_groups.append(
                await self._tools.search_knowledge(
                    SearchKnowledgeInput(query=query, limit=self._profile.search_limit)
                )
            )
        return hit_groups

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
        parsed, raw_response = await self._invoke_structured(
            EvidenceGrade,
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{GRADE_EVIDENCE}"),
                HumanMessage(
                    content=f"Question:\n{state['standalone_question']}\n\nEvidence:\n{_evidence_payload(state)}"
                ),
            ],
        )
        return {
            "evidence_sufficient": parsed.sufficient,
            "insufficiency_reason": parsed.reason,
            "search_queries": parsed.refined_queries[:MAX_REFINED_QUERIES],
            "model_call_count": model_call_count,
            **_usage_state_update(state, raw_response),
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
            raise RuntimeError("Evidence refinement reached execution without a query")
        return {"search_queries": queries[: self._profile.max_refined_queries]}

    async def generate_answer(self, state: AgentState) -> AgentState:
        if not state.get("evidence"):
            return {
                "draft_answer": INSUFFICIENT_EVIDENCE_ANSWER,
                "final_answer": INSUFFICIENT_EVIDENCE_ANSWER,
                "grounding_valid": True,
            }
        model_call_count = self._next_model_call_count(state)
        response = await self._answer_model.ainvoke(
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
        return {
            "draft_answer": answer,
            "model_call_count": model_call_count,
            **_usage_state_update(state, response),
        }

    async def verify_grounding(self, state: AgentState) -> AgentState:
        if not state.get("evidence"):
            return {"grounding_valid": True, "grounding_issues": []}
        evidence = [EvidenceItem.model_validate(item) for item in state["evidence"]]
        deterministic_issues = citation_issues(state["draft_answer"], evidence)
        if deterministic_issues:
            return {"grounding_valid": False, "grounding_issues": deterministic_issues}

        model_call_count = self._next_model_call_count(state)
        parsed, raw_response = await self._invoke_structured(
            GroundingVerdict,
            [
                SystemMessage(content=f"{SYSTEM_GROUNDING_RULES}\n\n{VERIFY_GROUNDING}"),
                HumanMessage(
                    content=(
                        f"Question:\n{state['standalone_question']}\n\n"
                        f"Draft:\n{state['draft_answer']}\n\nEvidence:\n{_evidence_payload(state)}"
                    )
                ),
            ],
        )
        return {
            "grounding_valid": parsed.valid,
            "grounding_issues": parsed.issues,
            "model_call_count": model_call_count,
            **_usage_state_update(state, raw_response),
        }

    def route_after_verify(self, state: AgentState) -> str:
        return "finalize" if state.get("grounding_valid") else "repair"

    async def repair_answer(self, state: AgentState) -> AgentState:
        model_call_count = self._next_model_call_count(state)
        response = await self._answer_model.ainvoke(
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
        return {
            "draft_answer": str(response.content),
            "model_call_count": model_call_count,
            **_usage_state_update(state, response),
        }

    async def reject_ungrounded_answer(self, state: AgentState) -> AgentState:
        del state
        return {
            "final_answer": (
                "I couldn't produce an answer that was fully supported by the knowledge base."
            )
        }

    async def finalize(self, state: AgentState) -> AgentState:
        answer = state.get("final_answer") or state["draft_answer"]
        response_sources = _response_sources_from_state(state)
        retrieved_artifact_ids = [
            EvidenceItem.model_validate(item).artifact_id for item in state.get("evidence", [])
        ]
        if not retrieved_artifact_ids and state.get("reuse_turn_id") is not None:
            retrieved_artifact_ids = _artifact_ids_for_prior_turn(
                state.get("history", []),
                reuse_turn_id=state.get("reuse_turn_id"),
            )
        turn = ConversationTurn(
            agent_run_id=state["agent_run_id"],
            question=state["question"],
            answer=hide_artifact_citations(answer),
            sources=response_sources,
            retrieved_artifact_ids=retrieved_artifact_ids,
        )
        agent_run_id = state["agent_run_id"]
        prior_turns = state.get("history", [])
        prior_turns = [
            prior_turn for prior_turn in prior_turns if prior_turn["agent_run_id"] != agent_run_id
        ]
        history = [*prior_turns, turn][-self._profile.max_history_turns :]
        return {
            "final_answer": answer,
            "response_sources": response_sources,
            "history": history,
        }
