"""Tiny shared numeric helpers for the evaluation reports.

One definition each of mean / percentile / optional-sum so the metric modules do not carry
private copies that can drift apart.
"""

from __future__ import annotations

import math
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Nearest-rank percentile; ``quantile`` in [0, 1]. ``None`` for an empty input."""

    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, math.ceil(quantile * len(ordered)) - 1))
    return float(ordered[index])


def sum_optional(values: Sequence[int | None]) -> int | None:
    """Sum, or ``None`` if any element is ``None`` (so a partial total never looks complete)."""

    if not values or any(value is None for value in values):
        return None
    return sum(value for value in values if value is not None)
