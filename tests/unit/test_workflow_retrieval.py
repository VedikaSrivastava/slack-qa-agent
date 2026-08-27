from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel

from knowledge_assistant.agent.citations import citation_issues
from knowledge_assistant.agent.graph import build_graph
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE, AgentProfile
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.workflow_nodes import (
    INSUFFICIENT_EVIDENCE_ANSWER,
    MAX_EVIDENCE_PAYLOAD_CHARS,
    EvidenceGrade,
    GroundedAnswerNodes,
    GroundingVerdict,
    ModelCallBudgetExceededError,
    _evidence_payload,
)
from knowledge_assistant.retrieval.models import (
    MAX_CONTEXT_CHARS,
    AccountLookupInput,
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
    ) -> None:
        self.search_results = search_results
        self.artifacts = artifacts
        self.account_evidence = account_evidence or []
        self.read_requests: list[ReadArtifactsInput] = []
        self.account_lookup_requests: list[AccountLookupInput] = []

    async def search_knowledge(self, request: SearchKnowledgeInput) -> list[SearchHit]:
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

    async def lookup_accounts(self, request: AccountLookupInput) -> list[EvidenceItem]:
        self.account_lookup_requests.append(request)
        return list(self.account_evidence)


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
) -> AgentState:
    return AgentState(
        search_queries=[query],
        evidence=[item.model_dump(mode="json") for item in evidence or []],
        retrieval_round_count=retrieval_round_count,
        tool_call_count=tool_call_count,
        model_call_count=model_call_count,
        account_lookup=account_lookup,
    )


def _apply_state_update(state: AgentState, result: AgentState) -> AgentState:
    return cast(AgentState, {**state, **result})


def _result_evidence(result: AgentState) -> list[EvidenceItem]:
    return [EvidenceItem.model_validate(item) for item in result["evidence"]]


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
        _state("query", account_lookup={"region": "North America West"})
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
        _state("query", account_lookup={"region": "North America West"})
    )

    assert [item.artifact_id for item in _result_evidence(result)] == [
        f"account-{index}" for index in range(12)
    ]


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


async def test_empty_evidence_is_model_refined_instead_of_repeating_queries() -> None:
    structured_model = Mock()
    structured_model.ainvoke = AsyncMock(
        return_value=EvidenceGrade(
            sufficient=False,
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


def test_ordinary_markdown_label_is_not_treated_as_an_artifact_citation() -> None:
    evidence = [_evidence("art_a")]

    issues = citation_issues("See [documentation]. Supported fact [art_a].", evidence)

    assert issues == []


def test_grouped_artifact_citations_are_recognized() -> None:
    evidence = [_evidence("art_a"), _evidence("art_b")]

    issues = citation_issues("Supported comparison [art_a, art_b].", evidence)

    assert issues == []


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
    assert "outage" not in result["final_answer"]


def test_serialized_prompt_evidence_has_its_own_character_budget() -> None:
    escaped_content = '\\"' * (MAX_CONTEXT_CHARS // 2)
    state = _state("query", evidence=[_evidence("art_a", content=escaped_content)])
    state["evidence"][0]["metadata"] = {"unused": "x" * MAX_CONTEXT_CHARS}

    payload = _evidence_payload(state)
    serialized_evidence = json.loads(payload)

    assert len(payload) <= MAX_EVIDENCE_PAYLOAD_CHARS
    assert "metadata" not in serialized_evidence[0]
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


def test_repaired_answer_is_reverified_once_before_finalization() -> None:
    tools = FakeRetrievalTools(search_results={}, artifacts={})
    graph = build_graph(_nodes(tools))
    edges = {(edge.source, edge.target) for edge in graph.get_graph().edges}

    assert ("repair_answer", "verify_repair") in edges
    assert ("verify_repair", "reject_ungrounded_answer") in edges
    assert ("reject_ungrounded_answer", "finalize") in edges
