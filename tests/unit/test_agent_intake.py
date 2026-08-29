from __future__ import annotations

from dataclasses import replace
from typing import Any, cast
from unittest.mock import AsyncMock, Mock

import pytest
from langchain_core.language_models import BaseChatModel
from pydantic import ValidationError

from knowledge_assistant.agent.models import QuestionDisposition
from knowledge_assistant.agent.processor import _response_from_state
from knowledge_assistant.agent.profiles import PRODUCTION_PROFILE
from knowledge_assistant.agent.retrieval_tools import KnowledgeRetrievalTools
from knowledge_assistant.agent.state import AgentState
from knowledge_assistant.agent.workflow_nodes import (
    CAPABILITY_ANSWER,
    GREETING_ANSWER,
    OUT_OF_SCOPE_ANSWER,
    GroundedAnswerNodes,
    RetrievalPlan,
    StandaloneQuestion,
    StructuredOutputValidationError,
)


def _nodes(model: object) -> GroundedAnswerNodes:
    return GroundedAnswerNodes(
        cast(BaseChatModel, model),
        cast(KnowledgeRetrievalTools, cast(Any, object())),
        PRODUCTION_PROFILE,
    )


async def test_simple_greeting_finishes_without_model_or_retrieval_work() -> None:
    model = Mock()
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-greeting",
        "question": "  Hey!!!  ",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
    }

    resolved = await nodes.resolve_question(state)
    completed = await nodes.finalize({**state, **resolved})

    assert resolved["question_disposition"] is QuestionDisposition.GREETING
    assert resolved["final_answer"] == GREETING_ANSWER
    assert nodes.route_after_resolution({**state, **resolved}) == "finalize"
    assert completed["history"] == [
        {
            "agent_run_id": "run-greeting",
            "question": "  Hey!!!  ",
            "answer": GREETING_ANSWER,
            "sources": [],
            "retrieved_artifact_ids": [],
        }
    ]
    model.with_structured_output.assert_not_called()


async def test_unclear_initial_question_returns_one_follow_up_without_searching() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(
        return_value=RetrievalPlan(
            disposition=QuestionDisposition.NEEDS_CLARIFICATION,
            clarification_question="Which customer or product do you mean",
        )
    )
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-unclear",
        "question": "Can you tell me more about it?",
        "standalone_question": "Can you tell me more about it?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    planned = await nodes.plan_retrieval(state)

    assert planned["question_disposition"] is QuestionDisposition.NEEDS_CLARIFICATION
    assert planned["final_answer"] == "Which customer or product do you mean?"
    assert planned["search_queries"] == []
    assert planned["account_lookup"] is None
    assert planned["model_call_count"] == 1
    assert nodes.route_after_plan({**state, **planned}) == "finalize"
    planner.ainvoke.assert_awaited_once()


async def test_out_of_scope_request_gets_code_owned_scope_response() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(
        return_value=RetrievalPlan(disposition=QuestionDisposition.OUT_OF_SCOPE)
    )
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-scope",
        "question": "Write a poem about the weather today.",
        "standalone_question": "Write a poem about the weather today.",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    planned = await nodes.plan_retrieval(state)

    assert planned["question_disposition"] is QuestionDisposition.OUT_OF_SCOPE
    assert planned["final_answer"] == OUT_OF_SCOPE_ANSWER
    assert planned["search_queries"] == []
    assert nodes.route_after_plan({**state, **planned}) == "finalize"


async def test_capability_question_gets_code_owned_response_without_searching() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(
        return_value=RetrievalPlan(disposition=QuestionDisposition.CAPABILITY_QUESTION)
    )
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-capability",
        "question": "What can you help me with?",
        "standalone_question": "What can you help me with?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    planned = await nodes.plan_retrieval(state)

    assert planned["question_disposition"] is QuestionDisposition.CAPABILITY_QUESTION
    assert planned["final_answer"] == CAPABILITY_ANSWER
    assert planned["search_queries"] == []
    assert nodes.route_after_plan({**state, **planned}) == "finalize"


def _retrieval_plan_validation_error() -> ValidationError:
    try:
        RetrievalPlan(disposition=QuestionDisposition.KNOWLEDGE_QUESTION, queries=[])
    except ValidationError as error:
        return error
    raise AssertionError("expected RetrievalPlan to reject an empty knowledge-question plan")


async def test_plan_retrieval_reasks_once_when_the_first_plan_fails_schema_validation() -> None:
    valid_plan = RetrievalPlan(
        disposition=QuestionDisposition.KNOWLEDGE_QUESTION, queries=["verdant bay patch window"]
    )
    planner = Mock()
    planner.ainvoke = AsyncMock(side_effect=[_retrieval_plan_validation_error(), valid_plan])
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-reask",
        "question": "For Verdant Bay, what is the approved patch window?",
        "standalone_question": "For Verdant Bay, what is the approved patch window?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    planned = await nodes.plan_retrieval(state)

    assert planner.ainvoke.await_count == 2
    assert planned["model_call_count"] == 2
    assert planned["question_disposition"] is QuestionDisposition.KNOWLEDGE_QUESTION
    assert planned["search_queries"] == ["verdant bay patch window"]


async def test_plan_retrieval_fails_after_one_invalid_schema_repair() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(side_effect=_retrieval_plan_validation_error())
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-fallback",
        "question": "Please email BlueHarbor's CEO an apology.",
        "standalone_question": "Please email BlueHarbor's CEO an apology.",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    with pytest.raises(
        StructuredOutputValidationError,
        match="remained invalid after one repair attempt",
    ) as raised:
        await nodes.plan_retrieval(state)

    assert planner.ainvoke.await_count == 2
    assert raised.value.model_call_count == 2


async def test_plan_retrieval_does_not_reask_past_the_model_call_budget() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(side_effect=_retrieval_plan_validation_error())
    model = Mock()
    model.with_structured_output.return_value = planner
    profile = replace(PRODUCTION_PROFILE, max_model_calls=1)
    nodes = GroundedAnswerNodes(
        cast(BaseChatModel, model),
        cast(KnowledgeRetrievalTools, cast(Any, object())),
        profile,
    )
    state: AgentState = {
        "agent_run_id": "run-budgeted-reask",
        "question": "What happened?",
        "standalone_question": "What happened?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    with pytest.raises(RuntimeError, match="model-call budget"):
        await nodes.plan_retrieval(state)

    planner.ainvoke.assert_awaited_once()


async def test_context_resolves_an_ambiguous_follow_up_before_intake_planning() -> None:
    resolver = Mock()
    resolver.ainvoke = AsyncMock(
        return_value=StandaloneQuestion(question="When does Acme's enterprise contract renew?")
    )
    planner = Mock()
    planner.ainvoke = AsyncMock(
        return_value=RetrievalPlan(
            disposition=QuestionDisposition.KNOWLEDGE_QUESTION,
            show_sources=True,
            queries=["Acme enterprise contract renewal date"],
        )
    )
    model = Mock()
    model.with_structured_output.side_effect = [resolver, planner]
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-follow-up",
        "question": "When does it renew?",
        "standalone_question": "When does it renew?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [
            {
                "agent_run_id": "run-earlier",
                "question": "What plan is Acme on?",
                "answer": "Acme is on the enterprise plan [account-acme].",
            }
        ],
        "model_call_count": 0,
    }

    resolved = await nodes.resolve_question(state)
    planned = await nodes.plan_retrieval({**state, **resolved})

    assert resolved["standalone_question"] == "When does Acme's enterprise contract renew?"
    assert planned["question_disposition"] is QuestionDisposition.KNOWLEDGE_QUESTION
    assert planned["show_sources"] is True
    assert planned["search_queries"] == ["Acme enterprise contract renewal date"]
    assert planned["model_call_count"] == 2
    assert nodes.route_after_plan({**state, **resolved, **planned}) == "retrieve"
    resolver_prompt = str(resolver.ainvoke.await_args.args[0][1].content)
    assert "What plan is Acme on?" in resolver_prompt
    assert "When does it renew?" in resolver_prompt


async def test_source_only_follow_up_uses_selected_prior_turn_without_retrieval() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(
        return_value=RetrievalPlan(
            disposition=QuestionDisposition.KNOWLEDGE_QUESTION,
            response_mode="sources_only",
            reuse_turn_id="run-earlier",
        )
    )
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-sources",
        "question": "Can you support that?",
        "standalone_question": "Can you support that?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [
            {
                "agent_run_id": "run-earlier",
                "question": "What happened?",
                "answer": "The rollout was paused.",
                "sources": [{"artifact_id": "art_a", "title": "Rollout notes"}],
                "retrieved_artifact_ids": ["art_a", "art_b"],
            }
        ],
        "model_call_count": 0,
    }

    planned = await nodes.plan_retrieval(state)

    assert planned["response_mode"] == "sources_only"
    assert planned["reuse_turn_id"] == "run-earlier"
    assert planned["show_sources"] is True
    assert planned["response_sources"] == [
        {"artifact_id": "art_a", "title": "Rollout notes", "score": None, "snippet": None}
    ]
    assert nodes.route_after_plan({**state, **planned}) == "finalize"


async def test_plan_rejects_a_prior_turn_identifier_not_in_checkpoint_history() -> None:
    planner = Mock()
    planner.ainvoke = AsyncMock(
        return_value=RetrievalPlan(
            disposition=QuestionDisposition.KNOWLEDGE_QUESTION,
            queries=["rollout"],
            reuse_turn_id="invented-run",
        )
    )
    model = Mock()
    model.with_structured_output.return_value = planner
    nodes = _nodes(model)
    state: AgentState = {
        "agent_run_id": "run-current",
        "question": "What changed since then?",
        "standalone_question": "What changed since then?",
        "question_disposition": QuestionDisposition.KNOWLEDGE_QUESTION,
        "history": [],
        "model_call_count": 0,
    }

    with pytest.raises(StructuredOutputValidationError, match="unavailable prior turn"):
        await nodes.plan_retrieval(state)


def test_retrieval_plan_rejects_queries_for_non_knowledge_dispositions() -> None:
    with pytest.raises(ValidationError, match="cannot trigger retrieval"):
        RetrievalPlan(
            disposition=QuestionDisposition.OUT_OF_SCOPE,
            queries=["weather today"],
        )


def test_clarification_response_is_not_mislabeled_as_insufficient_evidence() -> None:
    response = _response_from_state(
        {
            "final_answer": "Which customer do you mean?",
            "question_disposition": QuestionDisposition.NEEDS_CLARIFICATION,
            "evidence": [],
            "evidence_sufficient": False,
            "grounding_valid": True,
        }
    )

    assert response.disposition is QuestionDisposition.NEEDS_CLARIFICATION
    assert response.requires_user_input is True
    assert response.insufficient_evidence is False
    assert response.sources == []
