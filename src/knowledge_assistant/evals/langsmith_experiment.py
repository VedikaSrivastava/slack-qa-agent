"""Execution and persistence of bounded LangSmith benchmark experiments."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import anyio
from langsmith import Client, aevaluate, tracing_context

from knowledge_assistant.agent.processor import create_question_processor
from knowledge_assistant.agent.profiles import EVALUATOR_MODEL_NAME, AgentProfile
from knowledge_assistant.config import EvaluationSettings
from knowledge_assistant.evals.langsmith_dataset import (
    DATASET_NAME,
    DATASET_VERSION_TAG,
    sync_official_dataset,
)
from knowledge_assistant.evals.langsmith_evaluators import (
    action_budget_pass,
    create_reference_correctness_evaluator,
    deterministic_pass,
    source_recall,
)
from knowledge_assistant.evals.models import EvalCase
from knowledge_assistant.evals.protocols import ExperimentProtocol


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


async def _write_experiment_summary(output_path: Path, summary: dict[str, Any]) -> None:
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(
        json.dumps(summary, indent=2, default=str),
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

    dataset = sync_official_dataset(client, cases)
    run_nonce = uuid.uuid4().hex[:8]
    reference_correctness = create_reference_correctness_evaluator(settings)

    async with create_question_processor(settings, profile) as processor:

        async def answer_example(inputs: dict[str, Any]) -> dict[str, Any]:
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
            metadata={
                "agent_profile": profile.name,
                "dataset_digest": dataset["dataset_digest"],
            },
        ):
            results = await aevaluate(
                answer_example,
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
        "project_stats": _project_statistics(project),
        "saved_at": datetime.now(UTC).isoformat(),
    }
    await _write_experiment_summary(output_path, summary)
    return summary
