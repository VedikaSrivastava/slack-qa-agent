"""Local evaluation dataset helpers."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence

from knowledge_assistant.evals.models import EvalCase

# Deterministic suites under `cases/<name>.json`, in increasing breadth.
SUITE_CHOICES = ("smoke", "full", "derived", "multiturn")


def dataset_digest(cases: Sequence[EvalCase]) -> str:
    payload = json.dumps(
        [case.model_dump(mode="json") for case in cases],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
