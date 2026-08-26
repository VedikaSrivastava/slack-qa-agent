"""Bounded generation of review-required robustness candidates in LangSmith."""

from __future__ import annotations

import hashlib
import json
import uuid
from datetime import UTC, datetime
from typing import Any, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langsmith import Client, tracing_context
from pydantic import BaseModel, Field

from slack_qa_agent.config import ExperimentSettings
from slack_qa_agent.evals.langsmith_integration import DATASET_NAME, dataset_digest
from slack_qa_agent.evals.models import EvalCase

AUGMENTATION_DATASET_NAME = "slack-qa-agent-augmentation-candidates"
AUGMENTATION_DATASET_VERSION = "candidate-v1"
AUGMENTATION_MODEL_NAME = "gpt-5.6-terra"


class CandidateQuestion(BaseModel):
    transformation: Literal["paraphrase", "multi-turn"]
    prior_turns: list[str] = Field(default_factory=list, max_length=2)
    question: str = Field(min_length=1, max_length=8_000)


class CandidateBatch(BaseModel):
    candidates: list[CandidateQuestion] = Field(min_length=1, max_length=2)


def _candidate_id(seed_id: str, candidate: CandidateQuestion) -> str:
    payload = json.dumps(candidate.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    suffix = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
    return f"candidate-{seed_id}-{suffix}"


def build_candidate_case(seed: EvalCase, candidate: CandidateQuestion) -> EvalCase:
    """Inherit only human-verified labels; generated text changes the input surface."""

    return seed.model_copy(
        update={
            "id": _candidate_id(seed.id, candidate),
            "category": f"candidate-{candidate.transformation}",
            "prior_turns": candidate.prior_turns,
            "question": candidate.question,
        }
    )


async def generate_augmentation_candidates(
    *,
    client: Client,
    settings: ExperimentSettings,
    seeds: list[EvalCase],
    candidates_per_case: int,
) -> dict[str, object]:
    if candidates_per_case not in (1, 2):
        raise ValueError("candidates_per_case must be 1 or 2")
    source_digest = dataset_digest(seeds)
    if client.has_dataset(dataset_name=AUGMENTATION_DATASET_NAME):
        dataset = client.read_dataset(dataset_name=AUGMENTATION_DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=AUGMENTATION_DATASET_NAME,
            description=(
                "Model-generated robustness candidates derived from official seeds. "
                "Every example requires human review and must remain separate from the gold dataset."
            ),
            metadata={
                "source_dataset": DATASET_NAME,
                "review_required": True,
                "version": AUGMENTATION_DATASET_VERSION,
            },
        )

    generator = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=AUGMENTATION_MODEL_NAME,
    ).with_structured_output(CandidateBatch)
    examples: list[dict[str, Any]] = []
    with tracing_context(
        project_name=settings.langsmith_project,
        enabled=True,
        client=client,
        tags=["eval-augmentation", AUGMENTATION_DATASET_VERSION],
        metadata={
            "generator_model": AUGMENTATION_MODEL_NAME,
            "source_dataset_digest": source_digest,
        },
    ):
        for seed in seeds:
            generated = cast(
                CandidateBatch,
                await generator.ainvoke(
                    [
                        SystemMessage(
                            content=(
                                "Create evaluation-question variants without changing the requested "
                                "facts or answer. Use only paraphrase or a short multi-turn setup. Do "
                                "not include the answer, citations, hints, or new requirements. Return "
                                f"exactly {candidates_per_case} distinct candidates."
                            )
                        ),
                        HumanMessage(
                            content=json.dumps(
                                {
                                    "seed_question": seed.question,
                                    "reference_answer": seed.reference_answer,
                                    "allowed_transformations": ["paraphrase", "multi-turn"],
                                },
                                ensure_ascii=False,
                            )
                        ),
                    ]
                ),
            )
            if len(generated.candidates) != candidates_per_case:
                raise RuntimeError(
                    f"Augmentation model returned {len(generated.candidates)} candidates; "
                    f"expected exactly {candidates_per_case}"
                )
            for candidate in generated.candidates:
                candidate_case = build_candidate_case(seed, candidate)
                examples.append(
                    {
                        "id": uuid.uuid5(
                            uuid.NAMESPACE_URL,
                            f"{AUGMENTATION_DATASET_NAME}:{candidate_case.id}",
                        ),
                        "inputs": {
                            "case_id": candidate_case.id,
                            "seed_case_id": seed.id,
                            "question": candidate_case.question,
                            "prior_turns": candidate_case.prior_turns,
                        },
                        "outputs": {"case": candidate_case.model_dump(mode="json")},
                        "metadata": {
                            "review_status": "candidate",
                            "generator_model": AUGMENTATION_MODEL_NAME,
                            "source_dataset": DATASET_NAME,
                            "source_dataset_digest": source_digest,
                            "transformation": candidate.transformation,
                        },
                        "split": ["candidate", candidate.transformation],
                    }
                )

    client.create_examples(dataset_id=dataset.id, examples=examples, max_concurrency=1)
    tagged_at = datetime.now(UTC)
    client.update_dataset_tag(
        dataset_id=dataset.id,
        as_of=tagged_at,
        tag=AUGMENTATION_DATASET_VERSION,
    )
    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset.name,
        "dataset_version": AUGMENTATION_DATASET_VERSION,
        "source_dataset": DATASET_NAME,
        "source_dataset_digest": source_digest,
        "generator_model": AUGMENTATION_MODEL_NAME,
        "candidate_count": len(examples),
        "review_required": True,
    }
