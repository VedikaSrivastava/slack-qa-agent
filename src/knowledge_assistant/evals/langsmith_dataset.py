"""Versioned synchronization of the human-curated LangSmith gold dataset."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Iterable, Mapping
from datetime import datetime
from typing import Any

from langsmith import Client
from langsmith.schemas import Example
from langsmith.utils import LangSmithNotFoundError

from knowledge_assistant.evals.models import EvalCase

DATASET_NAME = "slack-qa-agent-official"
DATASET_DESCRIPTION = (
    "Human-curated gold cases from the Applied AI Slack Q&A assignment. "
    "Synthetic and production-derived cases must not be added to this dataset."
)
DATASET_VERSION_TAG = "official-v1"


class OfficialDatasetIntegrityError(RuntimeError):
    """Raised before the immutable official dataset can be changed or misidentified."""


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


def _assert_examples_match(
    actual_examples: Iterable[Example],
    expected_examples: list[dict[str, Any]],
) -> None:
    expected_by_id = {example["id"]: example for example in expected_examples}
    actual_by_id = {example.id: example for example in actual_examples}
    if set(actual_by_id) != set(expected_by_id):
        missing_ids = sorted(str(value) for value in set(expected_by_id) - set(actual_by_id))
        unexpected_ids = sorted(str(value) for value in set(actual_by_id) - set(expected_by_id))
        raise OfficialDatasetIntegrityError(
            "Official dataset example IDs do not match: "
            f"missing={missing_ids}, unexpected={unexpected_ids}"
        )

    for example_id, expected in expected_by_id.items():
        actual = actual_by_id[example_id]
        for field_name in ("inputs", "outputs", "metadata"):
            if getattr(actual, field_name) != expected[field_name]:
                raise OfficialDatasetIntegrityError(
                    f"Official dataset example {example_id} has unexpected {field_name}"
                )


def _validate_tagged_dataset(
    client: Client,
    *,
    dataset_id: uuid.UUID,
    expected_examples: list[dict[str, Any]],
) -> None:
    tagged_examples = list(client.list_examples(dataset_id=dataset_id, as_of=DATASET_VERSION_TAG))
    _assert_examples_match(tagged_examples, expected_examples)
    official_split_examples = list(
        client.list_examples(
            dataset_id=dataset_id,
            as_of=DATASET_VERSION_TAG,
            splits=["official"],
        )
    )
    _assert_examples_match(official_split_examples, expected_examples)


def sync_official_dataset(client: Client, cases: list[EvalCase]) -> dict[str, Any]:
    """Create the official tag once, then verify rather than move or rewrite it."""

    if client.has_dataset(dataset_name=DATASET_NAME):
        dataset = client.read_dataset(dataset_name=DATASET_NAME)
    else:
        dataset = client.create_dataset(
            dataset_name=DATASET_NAME,
            description=DATASET_DESCRIPTION,
            metadata={"source": "official-assignment", "version": DATASET_VERSION_TAG},
        )

    examples = [_build_example(case) for case in cases]
    try:
        client.read_dataset_version(dataset_id=dataset.id, tag=DATASET_VERSION_TAG)
    except LangSmithNotFoundError:
        has_official_version = False
    else:
        has_official_version = True

    if not has_official_version:
        current_examples = list(client.list_examples(dataset_id=dataset.id))
        expected_ids = {example["id"] for example in examples}
        unexpected_ids = sorted(
            str(example.id) for example in current_examples if example.id not in expected_ids
        )
        if unexpected_ids:
            raise OfficialDatasetIntegrityError(
                f"Official dataset contains unexpected example IDs: {unexpected_ids}"
            )

        upsert_result = client.create_examples(
            dataset_id=dataset.id,
            examples=examples,
            max_concurrency=1,
        )
        created_as_of = upsert_result.get("as_of")
        if not isinstance(created_as_of, str) or not created_as_of:
            raise OfficialDatasetIntegrityError(
                "LangSmith did not return the exact created dataset version"
            )
        # LangSmith tags are movable pointers, so assign the reviewed tag only once to an exact
        # server version. Future syncs verify this snapshot and never repoint it.
        client.update_dataset_tag(
            dataset_id=dataset.id,
            as_of=datetime.fromisoformat(created_as_of),
            tag=DATASET_VERSION_TAG,
        )

    _validate_tagged_dataset(client, dataset_id=dataset.id, expected_examples=examples)
    return {
        "dataset_id": str(dataset.id),
        "dataset_name": dataset.name,
        "dataset_version": DATASET_VERSION_TAG,
        "dataset_digest": dataset_digest(cases),
        "example_count": len(examples),
    }
