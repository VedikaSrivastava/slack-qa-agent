"""Versioned LangSmith dataset synchronization and experiment execution."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import anyio
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import Client, aevaluate, tracing_context
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Example, Run
from pydantic import BaseModel, Field

from slack_qa_agent.agent.models import AgentResponse
from slack_qa_agent.agent.profiles import EVALUATOR_MODEL_NAME, AgentProfile
from slack_qa_agent.agent.service import create_question_processor
from slack_qa_agent.config import ExperimentSettings
from slack_qa_agent.evals.evaluators import evaluate_response
from slack_qa_agent.evals.models import EvalCase
from slack_qa_agent.evals.protocols import ExperimentProtocol

DATASET_NAME = "slack-qa-agent-official"
DATASET_DESCRIPTION = (
    "Human-curated gold cases from the Applied AI Slack Q&A assignment. "
    "Synthetic and production-derived cases must not be added to this dataset."
)
DATASET_VERSION_TAG = "official-v1"


class ReferenceCorrectnessVerdict(BaseModel):
    """Semantic correctness verdict complementary to exact deterministic checks."""

    correct: bool
    score: float = Field(ge=0, le=1)
    rationale: str = Field(min_length=1, max_length=1_000)


def dataset_digest(cases: list[EvalCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _example_id(case_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{DATASET_NAME}:{case_id}")


def _reference_outputs(case: EvalCase) -> dict[str, Any]:
    return {"case": case.model_dump(mode="json")}


def sync_official_dataset(client: Client, cases: list[EvalCase]) -> dict[str, Any]:
    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
            metadata={"source": "official-assignment", "version": DATASET_VERSION_TAG},
        )

    examples = [
        {
            "id": _example_id(case.id),
            "inputs": {
                "case_id": case.id,
                "category": case.category,
                "question": case.question,
                "prior_turns": case.prior_turns,
            },
            "outputs": _reference_outputs(case),
            "metadata": {
                "case_id": case.id,
                "category": case.category,
                "source": "official-assignment",
                "dataset_version": DATASET_VERSION_TAG,
            },
            "split": ["official", case.category],
        }
        for case in cases
    ]
    client.create_examples(dataset_id=dataset.id, examples=examples, max_concurrency=1)
    tagged_at = datetime.now(UTC)
    client.update_dataset_tag(dataset_id=dataset.id, as_of=tagged_at, tag=DATASET_VERSION_TAG)
    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset.name,
        "dataset_version": DATASET_VERSION_TAG,
        "dataset_digest": dataset_digest(cases),
        "example_count": len(examples),
    }


def _case_from_reference(reference_outputs: Mapping[str, Any] | None) -> EvalCase:
    if reference_outputs is None or "case" not in reference_outputs:
        raise ValueError("LangSmith example is missing the reference EvalCase")
    return EvalCase.model_validate(reference_outputs["case"])


def deterministic_pass(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = _case_from_reference(reference_outputs)
    response = AgentResponse.model_validate(outputs["response"])
    result = evaluate_response(case, response)
    failures = [check.name for check in result.checks if not check.passed]
    return {
        "key": "deterministic_pass",
        "score": int(result.passed),
        "comment": "passed" if not failures else f"failed checks: {', '.join(failures)}",
    }


def source_recall(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = _case_from_reference(reference_outputs)
    expected = set(case.expected_source_ids)
    actual = {
        str(source["artifact_id"])
        for source in outputs["response"].get("sources", [])
        if isinstance(source, dict) and "artifact_id" in source
    }
    score = 1.0 if not expected else len(expected & actual) / len(expected)
    return {"key": "source_recall", "score": score}


def action_budget_pass(
    inputs: dict[str, Any],
    outputs: dict[str, Any],
    reference_outputs: Mapping[str, Any] | None,
) -> dict[str, Any]:
    del inputs
    case = _case_from_reference(reference_outputs)
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


async def run_langsmith_experiment(
    *,
    client: Client,
    settings: ExperimentSettings,
    cases: list[EvalCase],
    profile: AgentProfile,
    protocol: ExperimentProtocol,
    output_path: Path,
) -> dict[str, Any]:
    dataset = sync_official_dataset(client, cases)
    run_nonce = uuid.uuid4().hex[:8]
    evaluator_model = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=EVALUATOR_MODEL_NAME,
    ).with_structured_output(ReferenceCorrectnessVerdict)

    async def reference_correctness(
        run: Run,
        example: Example | None,
    ) -> EvaluationResult:
        if example is None or example.inputs is None:
            raise ValueError("Reference correctness requires a LangSmith example with inputs")
        case = _case_from_reference(example.outputs)
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

    async with create_question_processor(settings, profile) as processor:

        async def target(inputs: dict[str, Any]) -> dict[str, Any]:
            conversation_id = f"langsmith:{profile.name}:{inputs['case_id']}:{run_nonce}"
            for prior_turn in inputs.get("prior_turns", []):
                await processor.answer(
                    question=str(prior_turn),
                    conversation_id=conversation_id,
                    agent_run_id=str(uuid.uuid4()),
                )
            response = await processor.answer(
                question=str(inputs["question"]),
                conversation_id=conversation_id,
                agent_run_id=str(uuid.uuid4()),
            )
            return {"response": response.model_dump(mode="json")}

        with tracing_context(
            project_name=settings.langsmith_project,
            enabled=True,
            client=client,
            tags=["offline-eval", DATASET_VERSION_TAG, profile.name, protocol.name],
            metadata={"agent_profile": profile.name, "dataset_digest": dataset["dataset_digest"]},
        ):
            results = await aevaluate(
                target,
                data=client.list_examples(
                    dataset_name=DATASET_NAME,
                    as_of=DATASET_VERSION_TAG,
                    splits=["official"],
                ),
                evaluators=[
                    deterministic_pass,
                    source_recall,
                    action_budget_pass,
                    reference_correctness,
                ],
                experiment_prefix=f"slack-qa-{profile.name}-{protocol.name}",
                description="Official seven-question Slack Q&A benchmark.",
                metadata={
                    "models": [profile.model_name],
                    "evaluator_model": EVALUATOR_MODEL_NAME,
                    "agent_profile": profile.name,
                    "profile": asdict(profile),
                    "protocol": asdict(protocol),
                    "dataset_digest": dataset["dataset_digest"],
                    "dataset_version": DATASET_VERSION_TAG,
                },
                max_concurrency=protocol.max_concurrency,
                num_repetitions=protocol.repetitions,
                client=client,
                error_handling="log",
            )
            await results.wait()

    project = client.read_project(project_name=results.experiment_name, include_stats=True)
    summary = {
        **dataset,
        "experiment_id": str(results.experiment_id),
        "experiment_name": results.experiment_name,
        "experiment_url": results.url,
        "comparison_url": results.get_comparison_url(),
        "profile": asdict(profile),
        "protocol": asdict(protocol),
        "project_stats": {
            "run_count": project.run_count,
            "latency_p50": project.latency_p50,
            "latency_p99": project.latency_p99,
            "total_tokens": project.total_tokens,
            "prompt_tokens": project.prompt_tokens,
            "completion_tokens": project.completion_tokens,
            "total_cost": project.total_cost,
            "error_rate": project.error_rate,
            "feedback_stats": project.feedback_stats,
        },
        "saved_at": datetime.now(UTC).isoformat(),
    }
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary
