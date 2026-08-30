"""Local evaluation dataset helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from knowledge_assistant.evals.models import EvalCase

# Deterministic suites under `cases/<name>.json`, in increasing breadth.
SUITE_CHOICES = ("smoke", "full", "derived", "multiturn")
EVALUATION_PROTOCOL_VERSION = "v13"
# Assignment questions after removing the Slack mention prefix, plus the supplied answers.
TAKE_HOME_GOLD_DATASET_DIGEST = "ac16383cd1ad83a28b5a5225e3f15ded8bf983425f6f6baa1e217f0fd456ee1c"

_ANNOTATION_FIELDS = (
    "category",
    "expected_facts",
    "expected_entities",
    "expected_dates",
    "expected_commands",
    "expected_customers",
    "diagnostic_source_ids",
    "expected_show_sources",
    "forbidden_phrases",
    "max_tool_calls",
    "max_model_calls",
    "max_retrieval_rounds",
    "expected_insufficient_evidence",
    "insufficient_evidence_acceptable",
)


def _digest(payload: object) -> str:
    serialized = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def dataset_digest(cases: Sequence[EvalCase]) -> str:
    """Hash benchmark inputs separately from candidate-authored scoring annotations."""

    benchmark_cases: list[dict[str, object]] = []
    for case in cases:
        payload = case.model_dump(mode="json")
        for field in _ANNOTATION_FIELDS:
            payload.pop(field, None)
        benchmark_cases.append(payload)
    return _digest(benchmark_cases)


def annotation_digest(cases: Sequence[EvalCase]) -> str:
    """Hash lexical anchors, diagnostic source IDs, and operational budgets."""

    annotations: list[dict[str, object]] = []
    for case in cases:
        payload = case.model_dump(mode="json")
        annotations.append(
            {
                "id": case.id,
                **{field: payload[field] for field in _ANNOTATION_FIELDS if field in payload},
            }
        )
    return _digest(annotations)
