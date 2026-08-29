"""Aggregate one list of :class:`EvalResult` into the numbers the reports compare on.

Shared by the single-run CLI (``evals run``) and the multi-profile matrix so both report the
same fields with the same definitions: accuracy, per-check pass rates, action cost (tool and
model calls), token cost in dollars, latency percentiles, response length, and a small
reliability block.
"""

from __future__ import annotations

from knowledge_assistant.evals._stats import mean as _mean
from knowledge_assistant.evals._stats import percentile as _percentile
from knowledge_assistant.evals.models import EvalResult
from knowledge_assistant.evals.pricing import PRICES, PRICES_CAPTURED_AT, estimate_cost_usd

# Budget/quality checks whose failure means "the agent worked past its allowance or gave up",
# as opposed to simply getting a fact wrong.
_BUDGET_CHECK_NAMES = frozenset(
    {"tool_call_budget", "retrieval_round_budget", "evidence_sufficiency"}
)


def _failed_check_names(result: EvalResult) -> set[str]:
    return {check.name for check in result.checks if not check.passed}


def _split_model_cost_usd(
    classify_model: str,
    answer_model: str,
    *,
    input_tokens: int | None,
    output_tokens: int | None,
) -> float | None:
    """Approximate cost for a two-model profile.

    The harness tracks only whole-run token totals, not per-node. Input tokens are dominated by
    the evidence payload the structured classify/grade/verify steps carry; output tokens are
    dominated by the answer step. So price input at the classify model's input rate and output
    at the answer model's output rate. Labelled "split-approx" in the report.
    """

    classify_price = PRICES.get(classify_model)
    answer_price = PRICES.get(answer_model)
    if (
        classify_price is None
        or answer_price is None
        or input_tokens is None
        or output_tokens is None
    ):
        return None
    return (
        input_tokens * classify_price.input_per_mtok + output_tokens * answer_price.output_per_mtok
    ) / 1_000_000


def suite_metrics(
    results: list[EvalResult],
    *,
    model_name: str,
    answer_model_name: str | None = None,
) -> dict[str, object]:
    """Return the standard metric block for one profile's run over one suite.

    ``answer_model_name`` (when it differs from ``model_name``) switches the cost estimate to
    the two-model approximation described in ``_split_model_cost_usd``.
    """

    count = len(results)
    check_names = sorted({check.name for result in results for check in result.checks})
    check_pass_rates = {
        name: (
            sum(check.passed for result in results for check in result.checks if check.name == name)
            / sum(1 for result in results for check in result.checks if check.name == name)
        )
        for name in check_names
    }

    durations = [float(result.duration_ms) for result in results if result.duration_ms is not None]
    tool_calls = [result.tool_call_count for result in results]
    model_calls = [result.model_call_count for result in results]
    retrieval_rounds = [result.retrieval_round_count for result in results]
    answer_chars = [float(result.answer_chars) for result in results]
    answer_words = [float(result.answer_words) for result in results]

    # A completed case that made no model calls (a greeting short-circuit) reports token counts
    # of None, meaning "zero", not "unknown" -- coerce so one such case does not null out the
    # whole suite's token and cost totals.
    input_tokens = sum(result.input_tokens or 0 for result in results) if results else None
    output_tokens = sum(result.output_tokens or 0 for result in results) if results else None
    if answer_model_name is not None and answer_model_name != model_name:
        is_split = True
        total_cost = _split_model_cost_usd(
            model_name,
            answer_model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    else:
        is_split = False
        total_cost = estimate_cost_usd(
            model_name, input_tokens=input_tokens, output_tokens=output_tokens
        )

    budget_exceeded = sum(
        1 for result in results if _failed_check_names(result) & _BUDGET_CHECK_NAMES
    )
    insufficient = sum(
        1 for result in results if "evidence_sufficiency" in _failed_check_names(result)
    )

    # Fragment-level gold-label recall. `check_pass_rates["facts"]` is "every fragment matched"
    # and collapses to ~0 on multi-fragment cases; this is "fraction of fragments matched",
    # which ranks models even when none score a perfect case.
    hit_groups = sorted({group for result in results for group in result.deterministic_hits})
    label_hit_rates: dict[str, float | None] = {}
    for group in hit_groups:
        matched = sum(
            result.deterministic_hits[group].matched
            for result in results
            if group in result.deterministic_hits
        )
        total = sum(
            result.deterministic_hits[group].total
            for result in results
            if group in result.deterministic_hits
        )
        label_hit_rates[group] = matched / total if total else None
    all_matched = sum(
        hit.matched for result in results for hit in result.deterministic_hits.values()
    )
    all_total = sum(hit.total for result in results for hit in result.deterministic_hits.values())

    return {
        "case_count": count,
        "case_pass_rate": (sum(result.passed for result in results) / count if count else 0.0),
        "check_pass_rates": check_pass_rates,
        "label_hit_rates": label_hit_rates,
        "label_hit_rate_overall": all_matched / all_total if all_total else None,
        "per_case_pass": {result.case_id: result.passed for result in results},
        "latency_ms": {
            "p50": _percentile(durations, 0.50),
            "p95": _percentile(durations, 0.95),
            "p99": _percentile(durations, 0.99),
            "mean": _mean(durations),
            "max": max(durations) if durations else None,
        },
        "tool_calls": {
            "total": sum(tool_calls),
            "mean_per_case": _mean([float(value) for value in tool_calls]),
            "max": max(tool_calls) if tool_calls else None,
        },
        "model_calls": {
            "total": sum(model_calls),
            "mean_per_case": _mean([float(value) for value in model_calls]),
            "max": max(model_calls) if model_calls else None,
        },
        "retrieval_rounds": {
            "mean_per_case": _mean([float(value) for value in retrieval_rounds]),
            "max": max(retrieval_rounds) if retrieval_rounds else None,
        },
        "tokens": {
            "input_total": input_tokens,
            "output_total": output_tokens,
            "total": (
                input_tokens + output_tokens
                if input_tokens is not None and output_tokens is not None
                else None
            ),
            "mean_total_per_case": (
                (input_tokens + output_tokens) / count
                if input_tokens is not None and output_tokens is not None and count
                else None
            ),
        },
        "cost_usd": {
            "total": total_cost,
            "per_case": total_cost / count if total_cost is not None and count else None,
            "per_1k_cases": total_cost / count * 1_000
            if total_cost is not None and count
            else None,
            "method": "split-approx" if is_split else "single-model",
            "answer_model": answer_model_name if is_split else model_name,
            "prices_captured_at": PRICES_CAPTURED_AT,
        },
        "answer_length": {
            "chars_p50": _percentile(answer_chars, 0.50),
            "chars_mean": _mean(answer_chars),
            "chars_max": max(answer_chars) if answer_chars else None,
            "words_p50": _percentile(answer_words, 0.50),
            "words_mean": _mean(answer_words),
            "words_max": max(answer_words) if answer_words else None,
        },
        "reliability": {
            "passed_cases": sum(result.passed for result in results),
            "budget_exceeded_cases": budget_exceeded,
            "insufficient_evidence_cases": insufficient,
        },
    }
