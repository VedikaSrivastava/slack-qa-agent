"""Deterministic and model-graded metrics used by LangSmith experiments."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Mapping
from typing import Any, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.agent.profiles import (
    EVALUATOR_MODEL_NAME,
    OPENAI_MAX_RETRIES,
    OPENAI_REQUEST_TIMEOUT_SECONDS,
)
from knowledge_assistant.config import EvaluationSettings
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.langsmith_dataset import case_from_reference


class ReferenceCorrectnessVerdict(BaseModel):
    """Semantic correctness verdict complementary to exact deterministic checks."""

    correct: bool
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1_000)


def deterministic_pass(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = case_from_reference(reference_outputs)
    response = AgentResponse.model_validate(outputs["response"])
    result = evaluate_response(case, response)
    failures = [check.name for check in result.checks if not check.passed]
    return {
        "key": "deterministic_pass",
        "score": int(result.passed),
        "comment": "passed" if not failures else f"failed checks: {', '.join(failures)}",
    }


def _recall_score(expected_artifact_ids: set[str], actual_artifact_ids: set[str]) -> float:
    if not expected_artifact_ids:
        return 1.0
    return len(expected_artifact_ids & actual_artifact_ids) / len(expected_artifact_ids)


def citation_recall(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = case_from_reference(reference_outputs)
    response = AgentResponse.model_validate(outputs["response"])
    cited_artifact_ids = {source.artifact_id for source in response.sources}
    return {
        "key": "citation_recall",
        "score": _recall_score(set(case.expected_source_ids), cited_artifact_ids),
    }


def retrieval_recall(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = case_from_reference(reference_outputs)
    response = AgentResponse.model_validate(outputs["response"])
    return {
        "key": "retrieval_recall",
        "score": _recall_score(set(case.expected_source_ids), set(response.retrieved_artifact_ids)),
    }


def action_budget_pass(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = case_from_reference(reference_outputs)
    response = AgentResponse.model_validate(outputs["response"])
    passed = (
        response.tool_call_count <= case.max_tool_calls
        and response.retrieval_round_count <= case.max_retrieval_rounds
    )
    return {
        "key": "action_budget_pass",
        "score": int(passed),
        "comment": (
            f"tool_calls={response.tool_call_count}/{case.max_tool_calls}, "
            f"retrieval_rounds={response.retrieval_round_count}/{case.max_retrieval_rounds}"
        ),
    }


def create_reference_correctness_evaluator(
    settings: EvaluationSettings,
) -> Callable[[Run, Example | None], Awaitable[EvaluationResult]]:
    evaluator_model = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=EVALUATOR_MODEL_NAME,
        max_retries=OPENAI_MAX_RETRIES,
        timeout=OPENAI_REQUEST_TIMEOUT_SECONDS,
    ).with_structured_output(ReferenceCorrectnessVerdict)

    async def evaluate_reference_correctness(
        run: Run,
        example: Example | None,
    ) -> EvaluationResult:
        if example is None or example.inputs is None:
            raise ValueError("Reference correctness requires a LangSmith example with inputs")
        case = case_from_reference(example.outputs)
        response = AgentResponse.model_validate((run.outputs or {})["response"])
        verdict = cast(
            ReferenceCorrectnessVerdict,
            await evaluator_model.ainvoke(
                [
                    SystemMessage(
                        content=(
                            "Grade whether the candidate answer is factually correct and complete "
                            "for the question relative to the human reference. Allow harmless "
                            "paraphrase. Mark incorrect for missing requested parts, contradictions, "
                            "or invented facts. Do not reward style."
                        )
                    ),
                    HumanMessage(
                        content=json.dumps(
                            {
                                "question": example.inputs["question"],
                                "reference_answer": case.reference_answer,
                                "candidate_answer": response.answer,
                            },
                            ensure_ascii=False,
                        )
                    ),
                ]
            ),
        )
        return EvaluationResult(
            key="reference_correctness",
            score=verdict.score if verdict.correct else 0.0,
            comment=verdict.rationale,
        )

    return evaluate_reference_correctness
