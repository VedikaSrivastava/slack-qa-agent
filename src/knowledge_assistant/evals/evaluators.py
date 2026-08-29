"""Deterministic checks take precedence for exact expected values."""

from __future__ import annotations

import re

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.evals.models import CheckResult, EvalCase, EvalResult, HitCount


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9_./:-]+", " ", value.lower()).split())


def evaluate_response(
    case: EvalCase,
    response: AgentResponse,
    *,
    duration_ms: int | None = None,
) -> EvalResult:
    normalized_answer = _normalize(response.answer)
    checks: list[CheckResult] = []
    exact_groups = {
        "facts": case.expected_facts,
        "entities": case.expected_entities,
        "dates": case.expected_dates,
        "commands": case.expected_commands,
        "customers": case.expected_customers,
    }
    deterministic_hits: dict[str, HitCount] = {}
    for name, expected in exact_groups.items():
        missing = [value for value in expected if _normalize(value) not in normalized_answer]
        checks.append(
            CheckResult(
                name=name,
                passed=not missing,
                details=f"missing: {missing}" if missing else "",
            )
        )
        if expected:
            deterministic_hits[name] = HitCount(
                matched=len(expected) - len(missing), total=len(expected)
            )

    source_ids = [source.artifact_id for source in response.sources]
    missing_citations = sorted(set(case.expected_source_ids) - set(source_ids))
    missing_retrievals = sorted(
        set(case.expected_source_ids) - set(response.retrieved_artifact_ids)
    )
    forbidden_found = [
        value for value in case.forbidden_phrases if _normalize(value) in normalized_answer
    ]
    checks.extend(
        [
            CheckResult(
                name="source_attribution",
                passed=bool(source_ids) or case.insufficient_evidence_acceptable,
                details="no sources returned" if not source_ids else "",
            ),
            CheckResult(
                name="citation_recall",
                passed=not missing_citations,
                details=f"missing: {missing_citations}" if missing_citations else "",
            ),
            CheckResult(
                name="retrieval_recall",
                passed=not missing_retrievals,
                details=f"missing: {missing_retrievals}" if missing_retrievals else "",
            ),
            CheckResult(
                name="source_visibility",
                passed=(
                    case.expected_show_sources is None
                    or response.show_sources is case.expected_show_sources
                ),
                details=(
                    f"expected {case.expected_show_sources}, got {response.show_sources}"
                    if case.expected_show_sources is not None
                    and response.show_sources is not case.expected_show_sources
                    else ""
                ),
            ),
            CheckResult(
                name="forbidden_phrases",
                passed=not forbidden_found,
                details=f"found: {forbidden_found}" if forbidden_found else "",
            ),
            CheckResult(
                name="tool_call_budget",
                passed=response.tool_call_count <= case.max_tool_calls,
                details=(
                    f"{response.tool_call_count} > {case.max_tool_calls}"
                    if response.tool_call_count > case.max_tool_calls
                    else ""
                ),
            ),
            CheckResult(
                name="retrieval_round_budget",
                passed=response.retrieval_round_count <= case.max_retrieval_rounds,
                details=(
                    f"{response.retrieval_round_count} > {case.max_retrieval_rounds}"
                    if response.retrieval_round_count > case.max_retrieval_rounds
                    else ""
                ),
            ),
            CheckResult(
                name="model_call_budget",
                passed=(
                    case.max_model_calls is None
                    or response.model_call_count <= case.max_model_calls
                ),
                details=(
                    f"{response.model_call_count} > {case.max_model_calls}"
                    if case.max_model_calls is not None
                    and response.model_call_count > case.max_model_calls
                    else ""
                ),
            ),
            CheckResult(
                name="evidence_sufficiency",
                passed=not response.insufficient_evidence or case.insufficient_evidence_acceptable,
                details=(
                    "agent reported insufficient evidence"
                    if response.insufficient_evidence and not case.insufficient_evidence_acceptable
                    else ""
                ),
            ),
        ]
    )
    return EvalResult(
        case_id=case.id,
        passed=all(check.passed for check in checks),
        checks=checks,
        answer=response.answer,
        source_ids=source_ids,
        retrieved_artifact_ids=response.retrieved_artifact_ids,
        show_sources=response.show_sources,
        tool_call_count=response.tool_call_count,
        retrieval_round_count=response.retrieval_round_count,
        model_call_count=response.model_call_count,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
        duration_ms=duration_ms,
        answer_chars=len(response.answer),
        answer_words=len(response.answer.split()),
        deterministic_hits=deterministic_hits,
    )
