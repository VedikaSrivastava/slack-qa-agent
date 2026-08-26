"""Versioned synchronization of the human-curated LangSmith gold dataset."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any

from langsmith import Client

from knowledge_assistant.evals.models import EvalCase

DATASET_NAME = "slack-qa-agent-official"
DATASET_DESCRIPTION = (
    "Human-curated gold cases from the Applied AI Slack Q&A assignment. "
    "Synthetic and production-derived cases must not be added to this dataset."
)
DATASET_VERSION_TAG = "official-v1"


def dataset_digest(cases: list[EvalCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def case_from_reference(reference_outputs: Mapping[str, Any] | None) -> EvalCase:
    if reference_outputs is None or "case" not in reference_outputs:
        raise ValueError("LangSmith example is missing the reference EvalCase")
    return EvalCase.model_validate(reference_outputs["case"])


def _stable_example_id(case_id: str) -> uuid.UUID:
    return uuid.uuid5(uuid.NAMESPACE_URL, f"{DATASET_NAME}:{case_id}")


def _build_example(case: EvalCase) -> dict[str, Any]:
    return {
        "id": _stable_example_id(case.id),
        "inputs": {
            "case_id": case.id,
            "category": case.category,
            "question": case.question,
            "prior_turns": case.prior_turns,
        },
        "outputs": {"case": case.model_dump(mode="json")},
        "metadata": {
            "case_id": case.id,
            "category": case.category,
            "source": "official-assignment",
            "dataset_version": DATASET_VERSION_TAG,
        },
        "split": ["official", case.category],
    }


def sync_official_dataset(client: Client, cases: list[EvalCase]) -> dict[str, Any]:
    """Upsert the immutable official cases and tag their point-in-time version."""

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
            metadata={"source": "official-assignment", "version": DATASET_VERSION_TAG},
        )

    examples = [_build_example(case) for case in cases]
    client.create_examples(dataset_id=dataset.id, examples=examples, max_concurrency=1)
    client.update_dataset_tag(
        dataset_id=dataset.id,
        as_of=datetime.now(UTC),
        tag=DATASET_VERSION_TAG,
    )
    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset.name,
        "dataset_version": DATASET_VERSION_TAG,
        "dataset_digest": dataset_digest(cases),
        "example_count": len(examples),
    }
