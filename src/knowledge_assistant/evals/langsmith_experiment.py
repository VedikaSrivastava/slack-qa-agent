"""Execution and persistence of bounded LangSmith benchmark experiments."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from langsmith import Client, aevaluate, tracing_context

from knowledge_assistant.agent.processor import create_question_processor
from knowledge_assistant.agent.profiles import EVALUATOR_MODEL_NAME, AgentProfile
from knowledge_assistant.config import (
    APPLICATION_VERSION,
    LANGSMITH_PROJECT_NAME,
    PROMPT_VERSION,
    RETRIEVAL_VERSION,
    EvaluationSettings,
)
from knowledge_assistant.evals.langsmith_dataset import (
    DATASET_NAME,
    DATASET_VERSION_TAG,
    sync_official_dataset,
)
from knowledge_assistant.evals.langsmith_evaluators import (
    action_budget_pass,
    citation_recall,
    create_reference_correctness_evaluator,
    deterministic_pass,
    retrieval_recall,
)
from knowledge_assistant.evals.models import EvalCase
from knowledge_assistant.evals.protocols import ExperimentProtocol

EXPECTED_EVALUATOR_KEYS = frozenset(
    {
        "deterministic_pass",
        "citation_recall",
        "retrieval_recall",
        "action_budget_pass",
        "reference_correctness",
    }
)


def _project_statistics(project: Any) -> dict[str, Any]:
    return {
        "run_count": project.run_count,
        "latency_p50": project.latency_p50,
        "latency_p99": project.latency_p99,
        "total_tokens": project.total_tokens,
        "prompt_tokens": project.prompt_tokens,
        "completion_tokens": project.completion_tokens,
        "total_cost": project.total_cost,
        "error_rate": project.error_rate,
        "feedback_stats": project.feedback_stats,
    }


def require_error_free_experiment(project: Any, experiment_name: str) -> None:
    """Turn logged per-example execution errors into a failing experiment command."""

    if project.error_rate is None:
        raise RuntimeError(f"LangSmith experiment {experiment_name!r} did not report an error rate")
    if project.error_rate != 0.0:
        raise RuntimeError(
            f"LangSmith experiment {experiment_name!r} completed with "
            f"error_rate={project.error_rate}"
        )


def require_complete_experiment_results(
    result_rows: list[Mapping[str, Any]],
    *,
    expected_result_count: int,
    experiment_name: str,
) -> None:
    """Reject missing target runs or evaluator feedback before saving a successful summary."""

    if len(result_rows) != expected_result_count:
        raise RuntimeError(
            f"LangSmith experiment {experiment_name!r} returned {len(result_rows)} results; "
            f"expected {expected_result_count}"
        )

    for row_index, result_row in enumerate(result_rows):
        run = result_row.get("run")
        if run is None or getattr(run, "error", None):
            raise RuntimeError(
                f"LangSmith experiment {experiment_name!r} has a failed target run "
                f"at result {row_index}"
            )
        evaluation_results = result_row.get("evaluation_results")
        if not isinstance(evaluation_results, Mapping):
            raise RuntimeError(
                f"LangSmith experiment {experiment_name!r} is missing evaluator results "
                f"at result {row_index}"
            )
        feedback = evaluation_results.get("results")
        if not isinstance(feedback, list):
            raise RuntimeError(
                f"LangSmith experiment {experiment_name!r} is missing evaluator feedback "
                f"at result {row_index}"
            )
        feedback_keys = {
            key for result in feedback if isinstance(key := getattr(result, "key", None), str)
        }
        missing_keys = sorted(EXPECTED_EVALUATOR_KEYS - feedback_keys)
        if missing_keys:
            raise RuntimeError(
                f"LangSmith experiment {experiment_name!r} is missing evaluator keys "
                f"at result {row_index}: {missing_keys}"
            )


def _new_evaluation_conversation_id(profile_name: str, case_id: str) -> str:
    # `aevaluate` calls the target once per repetition. Allocating the ID here keeps each
    # repetition independent while preserving one thread across that repetition's prior turns.
    return f"langsmith:{profile_name}:{case_id}:{uuid.uuid4().hex}"


def _sanitize_experiment_summary(summary: dict[str, Any]) -> dict[str, Any]:
    """Remove private workspace links before a summary can be returned, printed, or persisted."""

    return {
        key: value
        for key, value in summary.items()
        if key not in {"experiment_url", "comparison_url"}
    }


async def _write_experiment_summary(output_path: Path, summary: dict[str, Any]) -> None:
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(
        json.dumps(_sanitize_experiment_summary(summary), indent=2, default=str),
        encoding="utf-8",
    )


async def run_official_benchmark(
    *,
    client: Client,
    settings: EvaluationSettings,
    cases: list[EvalCase],
    profile: AgentProfile,
    protocol: ExperimentProtocol,
    output_path: Path,
) -> dict[str, Any]:
    """Run one named agent profile against the versioned official dataset."""

    dataset = await anyio.to_thread.run_sync(sync_official_dataset, client, cases)
    reference_correctness = create_reference_correctness_evaluator(settings)

    async with create_question_processor(settings, profile) as processor:

        async def answer_example(inputs: dict[str, Any]) -> dict[str, Any]:
            conversation_id = _new_evaluation_conversation_id(profile.name, str(inputs["case_id"]))
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
            project_name=LANGSMITH_PROJECT_NAME,
            enabled=True,
            client=client,
            tags=["offline-eval", DATASET_VERSION_TAG, profile.name, protocol.name],
            metadata={
                "agent_profile": profile.name,
                "dataset_digest": dataset["dataset_digest"],
            },
        ):
            official_examples = await anyio.to_thread.run_sync(
                lambda: list(
                    client.list_examples(
                        dataset_name=DATASET_NAME,
                        as_of=DATASET_VERSION_TAG,
                        splits=["official"],
                    )
                )
            )
            results = await aevaluate(
                answer_example,
                data=official_examples,
                evaluators=[
                    deterministic_pass,
                    citation_recall,
                    retrieval_recall,
                    action_budget_pass,
                    reference_correctness,
                ],
                experiment_prefix=f"slack-qa-{profile.name}-{protocol.name}",
                description="Official seven-question Slack Q&A benchmark.",
                metadata={
                    "models": [profile.model_name],
                    "evaluator_model": EVALUATOR_MODEL_NAME,
                    "application_version": APPLICATION_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "retrieval_version": RETRIEVAL_VERSION,
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

    # LangSmith can log target/evaluator errors and still return a results object. Verify every
    # expected repetition explicitly before treating the experiment as complete.
    result_rows: list[Mapping[str, Any]] = [row async for row in results]
    require_complete_experiment_results(
        result_rows,
        expected_result_count=len(official_examples) * protocol.repetitions,
        experiment_name=results.experiment_name,
    )

    project = await anyio.to_thread.run_sync(
        lambda: client.read_project(project_name=results.experiment_name, include_stats=True)
    )
    require_error_free_experiment(project, results.experiment_name)
    summary = _sanitize_experiment_summary(
        {
            **dataset,
            "experiment_id": str(results.experiment_id),
            "experiment_name": results.experiment_name,
            "application_version": APPLICATION_VERSION,
            "prompt_version": PROMPT_VERSION,
            "retrieval_version": RETRIEVAL_VERSION,
            "evaluator_model": EVALUATOR_MODEL_NAME,
            "profile": asdict(profile),
            "protocol": asdict(protocol),
            "project_stats": _project_statistics(project),
            "saved_at": datetime.now(UTC).isoformat(),
        }
    )
    await _write_experiment_summary(output_path, summary)
    return summary
