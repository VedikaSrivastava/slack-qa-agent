from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from knowledge_assistant.agent.citations import citation_issues, hide_artifact_citations
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE, AgentProfile
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.workflow_nodes import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    MAX_EVIDENCE_PAYLOAD_CHARS,
    MAX_EVIDENCE_QUESTION_PART_CHARS,
    MAX_GROUNDING_ISSUE_CHARS,
    EvidenceGrade,
    GroundedAnswerNodes,
    GroundingVerdict,
    ModelCallBudgetExceededError,
    _evidence_payload,
    _restore_structured_customer_names,
)
from knowledge_assistant.retrieval.models import (
    MAX_ARTIFACT_BATCH,
    MAX_CONTEXT_CHARS,
    MAX_SEARCH_QUERY_CHARS,
    AccountLookupInput,
    AccountLookupResult,
    EvidenceItem,
    ReadArtifactsInput,
    SearchHit,
    SearchKnowledgeInput,
)


def _evidence(artifact_id: str, *, content: str | None = None) -> EvidenceItem:
    return EvidenceItem(
        artifact_id=artifact_id,
        title=f"Artifact {artifact_id}",
        snippet=f"Snippet {artifact_id}",
        content=content if content is not None else f"Evidence {artifact_id}",
    )


def _hit(artifact_id: str, score: float = 0.0) -> SearchHit:
    return SearchHit(
        artifact_id=artifact_id,
        title=f"Artifact {artifact_id}",
        snippet=f"Snippet {artifact_id}",
        score=score,
    )


class FakeRetrievalTools(KnowledgeRetrievalTools):
    def __init__(
        self,
        *,
        search_results: dict[str, list[SearchHit]],
        artifacts: dict[str, EvidenceItem],
        account_evidence: list[EvidenceItem] | None = None,
        account_matched_count: int | None = None,
    ) -> None:
        self.search_results = search_results
        self.artifacts = artifacts
        self.account_evidence = account_evidence or []
        self.account_matched_count = account_matched_count
        self.read_requests: list[ReadArtifactsInput] = []
        self.search_requests: list[SearchKnowledgeInput] = []
        self.account_lookup_requests: list[AccountLookupInput] = []

    async def search_knowledge(self, request: SearchKnowledgeInput) -> list[SearchHit]:
        self.search_requests.append(request)
        return list(self.search_results.get(request.query, []))

    async def read_artifacts(self, request: ReadArtifactsInput) -> list[EvidenceItem]:
        self.read_requests.append(request)
        remaining_context_chars = request.max_context_chars
        evidence: list[EvidenceItem] = []
        for artifact_id in request.artifact_ids:
            if remaining_context_chars <= 0:
                break
            item = self.artifacts[artifact_id]
            bounded_content = item.content[:remaining_context_chars]
            evidence.append(item.model_copy(update={"content": bounded_content}))
            remaining_context_chars -= len(bounded_content)
        return evidence

    async def lookup_accounts(self, request: AccountLookupInput) -> AccountLookupResult:
        self.account_lookup_requests.append(request)
        evidence = list(self.account_evidence[: request.limit])
        matched_account_count = (
            self.account_matched_count
            if self.account_matched_count is not None
            else len(self.account_evidence)
        )
        return AccountLookupResult(
            evidence=evidence,
            matched_account_count=matched_account_count,
            is_truncated=len(evidence) < matched_account_count,
        )


def _nodes(
    tools: FakeRetrievalTools, profile: AgentProfile = PRODUCTION_PROFILE
) -> GroundedAnswerNodes:
    unused_model = cast(BaseChatModel, cast(Any, object()))
    return GroundedAnswerNodes(unused_model, tools, profile)


def _state(
    query: str,
    *,
    evidence: list[EvidenceItem] | None = None,
    retrieval_round_count: int = 0,
    tool_call_count: int = 0,
    model_call_count: int = 0,
    account_lookup: dict[str, Any] | None = None,
    comparison_query: str | None = None,
) -> AgentState:
    return AgentState(
        search_queries=[query],
        evidence=[item.model_dump(mode="json") for item in evidence or []],
        retrieval_round_count=retrieval_round_count,
        tool_call_count=tool_call_count,
        model_call_count=model_call_count,
        account_lookup=account_lookup,
        comparison_query=comparison_query,
    )


def _apply_state_update(state: AgentState, result: AgentState) -> AgentState:
    return cast(AgentState, {**state, **result})


def _result_evidence(result: AgentState) -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(item) for item in result["evidence"]]


def test_structured_customer_names_restore_missing_spacing() -> None:
    evidence = _evidence(
        "account",
        content=json.dumps({"customer": "Maple Regional Transit Authority"}),
    ).model_copy(update={"retrieval_origin": "account_lookup"})

    answer = _restore_structured_customer_names(
        "Affected: MapleRegional Transit Authority.",
        [evidence],
    )

    assert answer == "Affected: Maple Regional Transit Authority."


def test_untrusted_json_artifact_cannot_trigger_customer_name_rewrite() -> None:
    evidence = _evidence(
        "lexical",
        content=json.dumps({"customer": "Maple Regional Transit Authority"}),
    )

    answer = _restore_structured_customer_names(
        "Affected: MapleRegional Transit Authority.",
        [evidence],
    )

    assert answer == "Affected: MapleRegional Transit Authority."


async def test_refinement_preserves_previous_evidence_and_adds_new_evidence() -> None:
    artifact_a = _evidence("A")
    artifact_b = _evidence("B")
    tools = FakeRetrievalTools(
        search_results={"first query": [_hit("A")], "refined query": [_hit("B")]},
        artifacts={"A": artifact_a, "B": artifact_b},
    )
    nodes = _nodes(tools)

    first_result = await nodes.execute_retrieval(_state("first query"))
    second_state = _apply_state_update(
        _state("refined query", retrieval_round_count=1),
        first_result,
    )
    second_state["search_queries"] = ["refined query"]
    second_result = await nodes.execute_retrieval(second_state)

    assert [item.artifact_id for item in _result_evidence(second_result)] == ["A", "B"]
    assert [request.artifact_ids for request in tools.read_requests] == [["A"], ["B"]]


async def test_comparative_retrieval_retains_bounded_excerpts_without_full_reads() -> None:
    candidate_hits = [_hit(f"candidate-{index}") for index in range(10)]
    tools = FakeRetrievalTools(
        search_results={"supplier launch risk": candidate_hits},
        artifacts={},
    )
    nodes = _nodes(tools)

    result = await nodes.execute_retrieval(
        _state("supplier milestone", comparison_query="supplier launch risk")
    )

    assert tools.search_requests == [
        SearchKnowledgeInput(
            query="supplier launch risk",
            purpose="comparison_candidates",
            limit=MAX_ARTIFACT_BATCH,
        )
    ]
    assert tools.read_requests == []
    evidence = _result_evidence(result)
    assert [item.artifact_id for item in evidence] == [
        f"candidate-{index}" for index in range(MAX_ARTIFACT_BATCH)
    ]
    assert all(item.retrieval_origin == "search_excerpt" for item in evidence)
    assert sum(len(item.content) for item in evidence) <= MAX_ARTIFACT_BATCH * 500
    assert result["tool_call_count"] == 1


async def test_refinement_promotes_a_candidate_excerpt_to_full_evidence() -> None:
    full_candidate = _evidence("candidate-a", content="Full candidate evidence")
    tools = FakeRetrievalTools(
        search_results={
            "supplier launch risk": [_hit("candidate-a"), _hit("candidate-b")],
            "candidate a deadline": [_hit("candidate-a")],
        },
        artifacts={"candidate-a": full_candidate},
    )
    nodes = _nodes(tools, replace(PRODUCTION_PROFILE, max_artifacts=2))
    initial_state = _state(
        "supplier deadline",
        comparison_query="supplier launch risk",
    )

    candidate_result = await nodes.execute_retrieval(initial_state)
    refinement_state = cast(AgentState, {**initial_state, **candidate_result})
    refinement_state["search_queries"] = ["candidate a deadline"]
    refined_result = await nodes.execute_retrieval(refinement_state)

    evidence = _result_evidence(refined_result)
    assert [item.artifact_id for item in evidence] == ["candidate-a", "candidate-b"]
    assert evidence[0].retrieval_origin == "lexical"
    assert evidence[0].content == "Full candidate evidence"
    assert evidence[1].retrieval_origin == "search_excerpt"
    assert [request.artifact_ids for request in tools.read_requests] == [["candidate-a"]]
    assert sum(len(item.content) for item in evidence) <= MAX_CONTEXT_CHARS


async def test_comparison_refinement_executes_all_preserved_queries() -> None:
    candidate_a = _evidence("candidate-a", content="Full evidence for candidate A")
    candidate_b = _evidence("candidate-b", content="Full evidence for candidate B")
    candidate_excerpts = [
        candidate_a.model_copy(update={"retrieval_origin": "search_excerpt"}),
        candidate_b.model_copy(update={"retrieval_origin": "search_excerpt"}),
    ]
    queries = [
        "supplier launch risk",
        "Candidate A milestone",
        "promised milestone details",
        "Candidate B milestone",
    ]
    tools = FakeRetrievalTools(
        search_results={
            "supplier launch risk": [_hit("candidate-a"), _hit("candidate-b")],
            "Candidate A milestone": [_hit("candidate-a")],
            "promised milestone details": [_hit("candidate-a"), _hit("candidate-b")],
            "Candidate B milestone": [_hit("candidate-b")],
        },
        artifacts={"candidate-a": candidate_a, "candidate-b": candidate_b},
    )
    nodes = _nodes(tools)
    state = _state(
        queries[0],
        evidence=candidate_excerpts,
        retrieval_round_count=1,
        comparison_query="supplier launch risk",
    )
    state["search_queries"] = queries

    result = await nodes.execute_retrieval(state)

    assert [request.query for request in tools.search_requests] == queries
    assert all(request.purpose == "evidence" for request in tools.search_requests)
    assert [request.artifact_ids for request in tools.read_requests] == [
        ["candidate-a", "candidate-b"]
    ]
    assert all(item.retrieval_origin == "lexical" for item in _result_evidence(result))
    assert result["tool_call_count"] == len(queries) + 1


async def test_ordinary_retrieval_keeps_profile_search_limit() -> None:
    tools = FakeRetrievalTools(search_results={"supplier status": []}, artifacts={})
    nodes = _nodes(tools)

    await nodes.execute_retrieval(_state("supplier status"))

    assert tools.search_requests == [
        SearchKnowledgeInput(query="supplier status", limit=PRODUCTION_PROFILE.search_limit)
    ]


async def test_refinement_deduplicates_hits_and_does_not_reread_existing_artifacts() -> None:
    artifact_a = _evidence("A")
    artifact_b = _evidence("B")
    tools = FakeRetrievalTools(
        search_results={"refined query": [_hit("A"), _hit("B")]},
        artifacts={"A": artifact_a, "B": artifact_b},
    )
    nodes = _nodes(tools)

    result = await nodes.execute_retrieval(
        _state(
            "refined query",
            evidence=[artifact_a],
            retrieval_round_count=1,
            tool_call_count=2,
        )
    )

    assert [item.artifact_id for item in _result_evidence(result)] == ["A", "B"]
    assert [request.artifact_ids for request in tools.read_requests] == [["B"]]


async def test_combined_evidence_respects_context_character_budget() -> None:
    artifact_a = _evidence("A", content="a" * (MAX_CONTEXT_CHARS - 1_200))
    artifact_b = _evidence("B", content="b" * 2_000)
    tools = FakeRetrievalTools(
        search_results={"refined query": [_hit("B")]},
        artifacts={"B": artifact_b},
    )
    nodes = _nodes(tools)

    result = await nodes.execute_retrieval(
        _state("refined query", evidence=[artifact_a], retrieval_round_count=1)
    )
    result_evidence = _result_evidence(result)

    assert sum(len(item.content) for item in result_evidence) == MAX_CONTEXT_CHARS
    assert len(result_evidence[1].content) == 1_200
    assert tools.read_requests[0].max_context_chars == 1_200


async def test_tool_call_accounting_remains_correct_across_refinement_rounds() -> None:
    tools = FakeRetrievalTools(
        search_results={"first query": [_hit("A")], "refined query": [_hit("A")]},
        artifacts={"A": _evidence("A")},
    )
    nodes = _nodes(tools)

    first_result = await nodes.execute_retrieval(_state("first query"))
    second_result = await nodes.execute_retrieval(
        _state(
            "refined query",
            evidence=_result_evidence(first_result),
            retrieval_round_count=1,
            tool_call_count=first_result["tool_call_count"],
        )
    )

    assert first_result["tool_call_count"] == 2  # one search and one artifact read
    assert second_result["tool_call_count"] == 3  # refinement searches but does not reread A
    assert len(tools.read_requests) == 1


async def test_structured_and_lexical_evidence_share_the_artifact_budget() -> None:
    profile = replace(PRODUCTION_PROFILE, max_artifacts=2)
    tools = FakeRetrievalTools(
        search_results={"query": [_hit("C")]},
        artifacts={"C": _evidence("C")},
        account_evidence=[_evidence("A"), _evidence("B"), _evidence("C")],
    )
    nodes = _nodes(tools, profile)

    result = await nodes.execute_retrieval(
        _state(
            "query",
            account_lookup={
                "purpose": "enumerate_cohort",
                "region": "North America West",
                "limit": 10,
            },
        )
    )

    assert [item.artifact_id for item in _result_evidence(result)] == ["A", "B"]
    assert tools.account_lookup_requests[0].limit == 2
    assert tools.read_requests == []
    assert result["tool_call_count"] == 1  # lexical work cannot improve a full artifact budget


async def test_production_profile_retains_all_twelve_structured_accounts() -> None:
    account_evidence = [_evidence(f"account-{index}") for index in range(12)]
    tools = FakeRetrievalTools(
        search_results={"query": []},
        artifacts={},
        account_evidence=account_evidence,
    )
    nodes = _nodes(tools)

    result = await nodes.execute_retrieval(
        _state(
            "query",
            account_lookup={
                "purpose": "enumerate_cohort",
                "region": "North America West",
                "limit": 10,
            },
        )
    )

    assert [item.artifact_id for item in _result_evidence(result)] == [
        f"account-{index}" for index in range(12)
    ]
    assert tools.account_lookup_requests[0].limit == PRODUCTION_PROFILE.max_artifacts


async def test_account_coverage_reflects_evidence_retained_after_merge() -> None:
    profile = replace(PRODUCTION_PROFILE, max_artifacts=3)
    tools = FakeRetrievalTools(
        search_results={"query": []},
        artifacts={},
        account_evidence=[_evidence("account-a"), _evidence("account-b")],
    )
    nodes = _nodes(tools, profile)

    result = await nodes.execute_retrieval(
        _state(
            "query",
            evidence=[_evidence("prior-a"), _evidence("prior-b")],
            account_lookup={
                "purpose": "enumerate_cohort",
                "region": "North America West",
                "limit": 10,
            },
        )
    )

    assert [item.artifact_id for item in _result_evidence(result)] == [
        "prior-a",
        "prior-b",
        "account-a",
    ]
    assert result["account_lookup_coverage"] == {
        "purpose": "enumerate_cohort",
        "matched_account_count": 2,
        "returned_account_count": 1,
        "is_truncated": True,
    }


async def test_larger_evidence_budget_does_not_increase_artifact_read_batch() -> None:
    artifact_ids = [f"artifact-{index}" for index in range(12)]
    tools = FakeRetrievalTools(
        search_results={"query": [_hit(artifact_id) for artifact_id in artifact_ids]},
        artifacts={artifact_id: _evidence(artifact_id) for artifact_id in artifact_ids},
    )
    nodes = _nodes(tools)

    result = await nodes.execute_retrieval(_state("query"))

    assert len(tools.read_requests[0].artifact_ids) == 8
    assert len(_result_evidence(result)) == 8


async def test_multiple_planned_queries_each_contribute_top_evidence() -> None:
    tools = FakeRetrievalTools(
        search_results={
            "competitor risk": [_hit("risk-first", -20), _hit("risk-second", -10)],
            "promised milestone": [_hit("milestone-first", -1), _hit("milestone-second", 0)],
        },
        artifacts={
            artifact_id: _evidence(artifact_id)
            for artifact_id in (
                "risk-first",
                "risk-second",
                "milestone-first",
                "milestone-second",
            )
        },
    )
    nodes = _nodes(tools, replace(PRODUCTION_PROFILE, max_artifacts=4))
    state = _state("competitor risk")
    state["search_queries"] = ["competitor risk", "promised milestone"]

    result = await nodes.execute_retrieval(state)

    assert [item.artifact_id for item in _result_evidence(result)] == [
        "risk-first",
        "milestone-first",
        "risk-second",
        "milestone-second",
    ]


async def test_contextual_follow_up_rereads_selected_prior_evidence_before_gap_search() -> None:
    tools = FakeRetrievalTools(
        search_results={"what changed": [_hit("new")]},
        artifacts={"prior": _evidence("prior"), "new": _evidence("new")},
    )
    nodes = _nodes(tools)
    state = _state("what changed")
    state["reuse_turn_id"] = "earlier"
    state["history"] = [
        {
            "agent_run_id": "earlier",
            "question": "What was planned?",
            "answer": "A rollout was planned.",
            "retrieved_artifact_ids": ["prior"],
        }
    ]

    result = await nodes.execute_retrieval(state)

    assert [item.artifact_id for item in _result_evidence(result)] == ["prior", "new"]
    assert [request.artifact_ids for request in tools.read_requests] == [["prior"], ["new"]]
    assert result["tool_call_count"] == 3


async def test_refinement_does_not_reread_selected_prior_evidence() -> None:
    tools = FakeRetrievalTools(
        search_results={"first": [], "refined": [_hit("new")]},
        artifacts={"prior": _evidence("prior"), "new": _evidence("new")},
    )
    nodes = _nodes(tools)
    first_state = _state("first")
    first_state["reuse_turn_id"] = "earlier"
    first_state["history"] = [
        {
            "agent_run_id": "earlier",
            "question": "What was planned?",
            "answer": "A rollout was planned.",
            "retrieved_artifact_ids": ["prior"],
        }
    ]

    first_result = await nodes.execute_retrieval(first_state)
    second_state = _apply_state_update(first_state, first_result)
    second_state["search_queries"] = ["refined"]
    second_result = await nodes.execute_retrieval(second_state)

    assert [item.artifact_id for item in _result_evidence(second_result)] == ["prior", "new"]
    assert [request.artifact_ids for request in tools.read_requests] == [["prior"], ["new"]]
    assert second_result["tool_call_count"] == 4


async def test_empty_evidence_is_model_refined_instead_of_repeating_queries() -> None:
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value=EvidenceGrade(
            supported_parts=[],
            missing_parts=["The requested fact"],
            reason="The first query found no evidence.",
            refined_queries=["materially different query"],
        )
    )
    model = Mock()
    model.with_structured_output.return_value = structured_model
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, PRODUCTION_PROFILE)
    state = _state("original query", retrieval_round_count=1)
    state["standalone_question"] = "Original question"

    result = await nodes.grade_evidence(state)

    assert result["evidence_sufficient"] is False
    assert result["search_queries"] == ["materially different query"]
    structured_model.ainvoke.assert_awaited_once()


async def test_comparison_grade_requires_full_evidence_and_preserves_follow_up_queries() -> None:
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value=EvidenceGrade(
            supported_parts=["Candidate A appears plausible from the shortlist."],
            missing_parts=[],
            reason="The shortlist appears sufficient.",
            refined_queries=["Candidate A milestone", "Candidate B milestone"],
        )
    )
    model = Mock()
    model.with_structured_output.return_value = structured_model
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, PRODUCTION_PROFILE)
    excerpt = _evidence("candidate-a", content="Candidate A has launch risk.").model_copy(
        update={
            "scenario_id": "scenario-a",
            "customer_name": "Candidate A",
            "retrieval_origin": "search_excerpt",
        }
    )
    state = _state(
        "promised milestone details",
        evidence=[excerpt],
        retrieval_round_count=1,
        comparison_query="supplier launch risk",
    )
    state["standalone_question"] = "Which supplier is riskiest, and what milestone follows?"

    result = await nodes.grade_evidence(state)

    assert result["evidence_sufficient"] is False
    assert result["search_queries"] == [
        "supplier launch risk",
        "Candidate A milestone",
        "promised milestone details",
        "Candidate B milestone",
    ]
    assert result["missing_question_parts"] == [
        "Comparison candidates still require full-artifact evidence before selection."
    ]
    messages = structured_model.ainvoke.await_args.args[0]
    prompt = cast(str, messages[1].content)
    assert '"customer_name":"Candidate A"' in prompt
    assert '"scenario_id":"scenario-a"' in prompt
    assert '"retrieval_origin":"search_excerpt"' in prompt
    assert "PLANNED_COMPARISON_FOLLOW_UP_QUERIES" in prompt
    assert '"promised milestone details"' in prompt


def test_evidence_grade_schema_exposes_the_question_part_character_limit() -> None:
    schema = EvidenceGrade.model_json_schema()

    assert (
        schema["properties"]["supported_parts"]["items"]["maxLength"]
        == MAX_EVIDENCE_QUESTION_PART_CHARS
    )
    assert (
        schema["properties"]["missing_parts"]["items"]["maxLength"]
        == MAX_EVIDENCE_QUESTION_PART_CHARS
    )
    assert schema["properties"]["refined_queries"]["items"]["maxLength"] == MAX_SEARCH_QUERY_CHARS


async def test_evidence_grade_reasks_once_after_question_part_validation_failure() -> None:
    with pytest.raises(ValidationError) as invalid_grade:
        EvidenceGrade(
            supported_parts=["x" * (MAX_EVIDENCE_QUESTION_PART_CHARS + 1)],
            missing_parts=[],
            reason="The evidence supports the answer.",
        )
    valid_grade = EvidenceGrade(
        supported_parts=["The requested value is directly supported."],
        missing_parts=[],
        reason="The evidence supports the answer.",
    )
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(side_effect=[invalid_grade.value, valid_grade])
    model = Mock()
    model.with_structured_output.return_value = structured_model
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, PRODUCTION_PROFILE)
    state = _state("query", evidence=[_evidence("art_a")])
    state["agent_run_id"] = "run-grade-reask"
    state["standalone_question"] = "What is the requested value?"

    result = await nodes.grade_evidence(state)

    assert structured_model.ainvoke.await_count == 2
    assert result["model_call_count"] == 2
    assert result["supported_question_parts"] == ["The requested value is directly supported."]
    assert result["evidence_sufficient"] is True


async def test_unknown_citation_fails_before_model_grounding_check() -> None:
    model = Mock()
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, PRODUCTION_PROFILE)
    state = _state("query", evidence=[_evidence("art_a")])
    state["draft_answer"] = "Unsupported answer [art_b]."

    result = await nodes.verify_grounding(state)

    assert result["grounding_valid"] is False
    assert "not retrieved" in result["grounding_issues"][0]
    model.with_structured_output.assert_not_called()


def test_grounding_verdict_schema_exposes_the_issue_character_limit() -> None:
    schema = GroundingVerdict.model_json_schema()

    assert schema["properties"]["issues"]["items"]["maxLength"] == MAX_GROUNDING_ISSUE_CHARS


async def test_grounding_verification_reasks_once_after_schema_validation_failure() -> None:
    with pytest.raises(ValidationError) as invalid_verdict:
        GroundingVerdict(valid=False, issues=[])
    valid_verdict = GroundingVerdict(valid=True)
    verifier = Mock()
    verifier.ainvoke = AsyncMock(side_effect=[invalid_verdict.value, valid_verdict])
    model = Mock()
    model.with_structured_output.return_value = verifier
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, PRODUCTION_PROFILE)
    state = _state("query", evidence=[_evidence("art_a")])
    state["agent_run_id"] = "run-verify-reask"
    state["standalone_question"] = "What happened?"
    state["draft_answer"] = "Supported answer [art_a]."

    result = await nodes.verify_grounding(state)

    assert verifier.ainvoke.await_count == 2
    assert result["model_call_count"] == 2
    assert result["grounding_valid"] is True


def test_ordinary_markdown_label_is_not_treated_as_an_artifact_citation() -> None:
    evidence = [_evidence("art_a")]

    issues = citation_issues("See [documentation]. Supported fact [art_a].", evidence)

    assert issues == []


def test_grouped_artifact_citations_are_recognized() -> None:
    evidence = [_evidence("art_a"), _evidence("art_b")]

    issues = citation_issues("Supported comparison [art_a, art_b].", evidence)

    assert issues == []


def test_hiding_provenance_preserves_ordinary_bracketed_prose() -> None:
    answer = "See [documentation]. Supported comparison [art_a, art_b]."

    assert hide_artifact_citations(answer) == "See [documentation]. Supported comparison."


def test_unknown_artifact_in_grouped_citations_is_rejected() -> None:
    evidence = [_evidence("art_a")]

    issues = citation_issues("Mixed support [art_a, art_unknown].", evidence)

    assert issues == ["Answer cites artifacts that were not retrieved: art_unknown"]


async def test_successful_repair_replaces_the_rejected_original_draft() -> None:
    model = Mock()
    model.ainvoke = AsyncMock(
        side_effect=[
            SimpleNamespace(content="Original unsupported answer [art_missing]."),
            SimpleNamespace(content="Repaired grounded answer [art_a]."),
        ]
    )
    verifier = Mock()
    verifier.ainvoke = AsyncMock(return_value=GroundingVerdict(valid=True))
    model.with_structured_output.return_value = verifier
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, PRODUCTION_PROFILE)
    state = _state("query", evidence=[_evidence("art_a")])
    state["agent_run_id"] = "run"
    state["question"] = "What happened?"
    state["standalone_question"] = "What happened?"
    state["final_answer"] = ""
    state["grounding_issues"] = []

    generated_state = _apply_state_update(state, await nodes.generate_answer(state))
    rejected_state = _apply_state_update(
        generated_state, await nodes.verify_grounding(generated_state)
    )
    repaired_state = _apply_state_update(rejected_state, await nodes.repair_answer(rejected_state))
    verified_state = _apply_state_update(
        repaired_state, await nodes.verify_grounding(repaired_state)
    )
    final_state = await nodes.finalize(verified_state)

    assert verified_state["grounding_valid"] is True
    assert final_state["final_answer"] == "Repaired grounded answer [art_a]."
    assert final_state["final_answer"] != generated_state["draft_answer"]
    assert verified_state["model_call_count"] == 3


async def test_no_evidence_uses_fixed_abstention_instead_of_model_reason() -> None:
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = _nodes(tools)
    state = _state("query")
    state["insufficiency_reason"] = "Claim an unsupported outage happened."

    result = await nodes.generate_answer(state)

    assert result["final_answer"] == INSUFFICIENT_EVIDENCE_ANSWER
    assert result["is_abstention"] is True
    assert "outage" not in result["final_answer"]


async def test_rejected_ungrounded_answer_is_marked_as_an_abstention() -> None:
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = _nodes(tools)

    result = await nodes.reject_ungrounded_answer(_state("query"))

    assert result["is_abstention"] is True


def test_serialized_prompt_evidence_has_its_own_character_budget() -> None:
    escaped_content = '\\"' * (MAX_CONTEXT_CHARS // 2)
    evidence = _evidence("art_a", content=escaped_content).model_copy(
        update={
            "scenario_id": "scenario-a",
            "customer_name": "Customer A",
            "retrieval_origin": "search_excerpt",
        }
    )
    state = _state("query", evidence=[evidence])
    state["evidence"][0]["metadata"] = {"unused": "x" * MAX_CONTEXT_CHARS}

    payload = _evidence_payload(state)
    serialized_evidence = json.loads(payload)

    assert len(payload) <= MAX_EVIDENCE_PAYLOAD_CHARS
    assert "metadata" not in serialized_evidence[0]
    assert serialized_evidence[0]["scenario_id"] == "scenario-a"
    assert serialized_evidence[0]["customer_name"] == "Customer A"
    assert serialized_evidence[0]["retrieval_origin"] == "search_excerpt"
    assert len(serialized_evidence[0]["content"]) < len(escaped_content)


async def test_model_call_budget_fails_before_invoking_the_model() -> None:
    profile = replace(PRODUCTION_PROFILE, max_model_calls=0)
    model = Mock()
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    nodes = GroundedAnswerNodes(cast(BaseChatModel, model), tools, profile)
    state = _state("query")
    state["standalone_question"] = "Question"

    with pytest.raises(ModelCallBudgetExceededError, match="model-call budget"):
        await nodes.plan_retrieval(state)

    model.with_structured_output.assert_not_called()
