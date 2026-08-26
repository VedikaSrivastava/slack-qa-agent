from knowledge_assistant.agent.models import AgentResponse, EvidenceReference
from knowledge_assistant.evals.evaluators import evaluate_response
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
        tool_call_count=3,
        retrieval_round_count=1,
    )

    result = evaluate_response(case, response)

    assert result.passed is True
