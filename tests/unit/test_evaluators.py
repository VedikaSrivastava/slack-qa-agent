from unittest.mock import patch

from knowledge_assistant.agent.models import AgentResponse, EvidenceReference
from knowledge_assistant.agent.profiles import (
    EVALUATOR_MODEL_NAME,
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
)
from knowledge_assistant.config import EvaluationSettings
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.langsmith_evaluators import (
    ReferenceCorrectnessVerdict,
    create_reference_correctness_evaluator,
)
from knowledge_assistant.evals.models import EvalCase


def test_exact_evaluator_checks_facts_sources_and_budgets() -> None:
    case = EvalCase(
        id="case",
        category="exact",
        question="When?",
        reference_answer="The date is 2026-08-25.",
        expected_dates=["2026-08-25"],
        expected_source_ids=["a1"],
        max_tool_calls=3,
    )
    response = AgentResponse(
        answer="The date is 2026-08-25 [a1].",
        sources=[EvidenceReference(artifact_id="a1", title="Runbook")],
        retrieved_artifact_ids=["a1"],
        tool_call_count=3,
        retrieval_round_count=1,
        model_call_count=4,
        input_tokens=120,
        output_tokens=30,
    )

    result = evaluate_response(case, response, duration_ms=250)

    assert result.passed is True
    assert result.source_ids == ["a1"]
    assert result.retrieved_artifact_ids == ["a1"]
    assert result.model_call_count == 4
    assert result.input_tokens == 120
    assert result.output_tokens == 30
    assert result.duration_ms == 250
    assert all(check.details == "" for check in result.checks)


def test_retrieval_and_citation_failures_are_reported_separately() -> None:
    case = EvalCase(
        id="case",
        category="retrieval",
        question="What happened?",
        reference_answer="Both artifacts are required.",
        expected_source_ids=["a1", "a2"],
    )
    response = AgentResponse(
        answer="Only the first fact [a1].",
        sources=[EvidenceReference(artifact_id="a1", title="First")],
        retrieved_artifact_ids=["a1", "a2"],
    )

    result = evaluate_response(case, response)
    checks = {check.name: check for check in result.checks}

    assert checks["retrieval_recall"].passed is True
    assert checks["citation_recall"].passed is False
    assert checks["citation_recall"].details == "missing: ['a2']"


def test_reference_evaluator_model_requests_are_bounded() -> None:
    settings = EvaluationSettings(
        _env_file=None,
        openai_api_key="test-openai-key",
        langsmith_api_key="test-langsmith-key",
        database_url="postgresql+asyncpg://user:password@postgres/test",
    )

    with patch("knowledge_assistant.evals.langsmith_evaluators.ChatOpenAI") as chat_model:
        create_reference_correctness_evaluator(settings)

    chat_model.assert_called_once_with(
        api_key=settings.openai_api_key,
        model=EVALUATOR_MODEL_NAME,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
    )
    chat_model.return_value.with_structured_output.assert_called_once_with(
        ReferenceCorrectnessVerdict
    )
