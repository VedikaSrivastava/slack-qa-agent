"""Static OpenAI list prices for turning token counts into a comparable dollar figure.

These are standard short-context USD prices per 1,000,000 tokens from the public OpenAI
model and pricing pages, captured on 2026-08-29. They are only used to rank profiles in
offline experiments; they are not a
billing source and will drift. Update ``PRICES`` (and ``PRICES_CAPTURED_AT``) when re-running
a cost-sensitive comparison. Reasoning models bill their hidden reasoning tokens as output
tokens, so their ``output`` rate dominates the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

PRICES_CAPTURED_AT = "2026-08-29"


@dataclass(frozen=True)
class TokenPrice:
    """USD per 1,000,000 tokens."""

    input_per_mtok: float
    output_per_mtok: float


# Keyed by the exact ``model_name`` used in an ``AgentProfile``.
PRICES: dict[str, TokenPrice] = {
    "gpt-4.1": TokenPrice(2.00, 8.00),
    "gpt-4.1-mini": TokenPrice(0.40, 1.60),
    "gpt-5": TokenPrice(1.25, 10.00),
    "gpt-5.5": TokenPrice(5.00, 30.00),
    "gpt-5.6-sol": TokenPrice(4.00, 20.00),
    "gpt-5.6-terra": TokenPrice(2.00, 12.00),
    "gpt-5.6-luna": TokenPrice(0.20, 1.20),
}


def estimate_cost_usd(
    model_name: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Best-effort dollar estimate for one profile's token totals.

    Returns ``None`` when either the price or a token count is unknown, so callers can render
    "n/a" instead of a misleading ``0.0``.
    """

    price = PRICES.get(model_name)
    if price is None or input_tokens is None or output_tokens is None:
        return None
    return (input_tokens * price.input_per_mtok + output_tokens * price.output_per_mtok) / 1_000_000
