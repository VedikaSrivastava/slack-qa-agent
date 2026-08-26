"""Deterministic checks take precedence for exact expected values."""

from __future__ import annotations

import re

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.evals.models import CheckResult, EvalCase, EvalResult


def _normalize(value: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9_./:-]+", " ", value.lower()).split())


def evaluate_response(case: EvalCase, response: AgentResponse) -> EvalResult:
    normalized_answer = _normalize(response.answer)
    checks: list[CheckResult] = []
    exact_groups = {
        "facts": case.expected_facts,
        "entities": case.expected_entities,
        "dates": case.expected_dates,
        "commands": case.expected_commands,
        "customers": case.expected_customers,
    }
    for name, expected in exact_groups.items():
        missing = [value for value in expected if _normalize(value) not in normalized_answer]
        checks.append(
            CheckResult(
                name=name,
                passed=not missing,
                details=f"missing: {missing}" if missing else "",
            )
        )

    source_ids = [source.artifact_id for source in response.sources]
    missing_sources = sorted(set(case.expected_source_ids) - set(source_ids))
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
                name="expected_source_ids",
                passed=not missing_sources,
                details=f"missing: {missing_sources}" if missing_sources else "",
            ),
            CheckResult(
                name="forbidden_phrases",
                passed=not forbidden_found,
                details=f"found: {forbidden_found}" if forbidden_found else "",
            ),
            CheckResult(
                name="tool_call_budget",
                passed=response.tool_call_count <= case.max_tool_calls,
                details=f"{response.tool_call_count} > {case.max_tool_calls}",
            ),
            CheckResult(
                name="retrieval_round_budget",
                passed=response.retrieval_round_count <= case.max_retrieval_rounds,
                details=f"{response.retrieval_round_count} > {case.max_retrieval_rounds}",
            ),
            CheckResult(
                name="evidence_sufficiency",
                passed=not response.insufficient_evidence or case.insufficient_evidence_acceptable,
                details="agent reported insufficient evidence",
            ),
        ]
    )
    return EvalResult(
        case_id=case.id,
        passed=all(check.passed for check in checks),
        checks=checks,
        answer=response.answer,
        source_ids=source_ids,
        tool_call_count=response.tool_call_count,
        retrieval_round_count=response.retrieval_round_count,
    )
