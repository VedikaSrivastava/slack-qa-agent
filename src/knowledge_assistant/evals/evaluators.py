"""Deterministic evaluation diagnostics and runtime-contract checks.

Free-form answer correctness is intentionally not decided here. Lexical anchors are useful for
debugging, but semantic paraphrases require the separate reference-based judge.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Sequence
from datetime import date
from typing import Literal

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.evals.models import (
    CheckResult,
    CheckScope,
    EvalCase,
    EvalResult,
    HitCount,
)

_ISO_DATE = re.compile(r"^(?P<year>\d{4})-(?P<month>\d{2})-(?P<day>\d{2})$")
_TIME_RANGE = re.compile(
    r"(?P<start>\b\d{1,2}:\d{2})\s*(?:-|to|through)\s*(?P<end>\d{1,2}:\d{2}\b)",
    flags=re.IGNORECASE,
)
_MONTH_NAMES = (
    "january",
    "february",
    "march",
    "april",
    "may",
    "june",
    "july",
    "august",
    "september",
    "october",
    "november",
    "december",
)
type _ContentMatchKind = Literal["lexical", "date", "command", "customer"]


def _normalize(value: str) -> str:
    normalized = unicodedata.normalize("NFKC", value).lower()
    normalized = normalized.replace("%", " percent ")
    normalized = normalized.replace("\N{EN DASH}", "-").replace("\N{EM DASH}", "-")
    normalized = _TIME_RANGE.sub(r"\g<start>-\g<end>", normalized)
    return " ".join(re.sub(r"[^a-z0-9_./:<>=-]+", " ", normalized).split())


def _date_variants(value: str) -> set[str]:
    match = _ISO_DATE.fullmatch(value)
    if match is None:
        return {_normalize(value)}
    parsed = date(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )
    month_name = _MONTH_NAMES[parsed.month - 1]
    month_abbreviation = month_name[:3]
    day = str(parsed.day)
    return {
        _normalize(value),
        _normalize(f"{month_name} {day} {parsed.year}"),
        _normalize(f"{month_abbreviation} {day} {parsed.year}"),
        _normalize(f"{parsed.month}/{parsed.day}/{parsed.year}"),
    }


def _contains_normalized_fragment(normalized_answer: str, fragment: str) -> bool:
    return (
        re.search(
            rf"(?<![a-z0-9_]){re.escape(fragment)}(?![a-z0-9_])",
            normalized_answer,
        )
        is not None
    )


def _is_date_present(
    expected: str,
    normalized_answer: str,
    normalized_question: str,
) -> bool:
    full_variants = _date_variants(expected)
    if any(_contains_normalized_fragment(normalized_answer, value) for value in full_variants):
        return True

    match = _ISO_DATE.fullmatch(expected)
    if match is None:
        return _contains_normalized_fragment(normalized_answer, _normalize(expected))

    parsed = date(
        int(match.group("year")),
        int(match.group("month")),
        int(match.group("day")),
    )
    month_name = _MONTH_NAMES[parsed.month - 1]
    month_abbreviation = month_name[:3]
    day = str(parsed.day)
    yearless_variants = {
        _normalize(f"{month_name} {day}"),
        _normalize(f"{month_abbreviation} {day}"),
    }
    if not any(
        _contains_normalized_fragment(normalized_answer, value) for value in yearless_variants
    ):
        return False

    if not _contains_normalized_fragment(normalized_question, str(parsed.year)):
        return False

    # A yearless answer is acceptable when the question itself supplies the year. An explicitly
    # conflicting year must not receive credit just because "March 23" is a substring.
    conflicting_named_date = re.search(
        rf"(?<![a-z0-9_])(?:{month_name}|{month_abbreviation}) {day} "
        rf"(?!{parsed.year}(?![0-9]))[0-9]{{4}}(?![a-z0-9_])",
        normalized_answer,
    )
    return conflicting_named_date is None


def _is_exact_command_present(expected: str, answer: str) -> bool:
    normalized_expected = " ".join(unicodedata.normalize("NFKC", expected).split())
    inline_commands = re.findall(r"`([^`\r\n]+)`", answer)
    inline_commands.extend(content for _, content in re.findall(r"(['\"])([^'\"\r\n]+)\1", answer))
    if any(
        " ".join(unicodedata.normalize("NFKC", command).split()) == normalized_expected
        for command in inline_commands
    ):
        return True

    # Do not let a valid command embedded inside a different code span (for example prefixed with
    # `sudo` or suffixed with `--force`) pass through the prose fallback below.
    answer_without_inline_commands = re.sub(r"`[^`\r\n]+`", "", answer)
    answer_without_inline_commands = re.sub(
        r"(['\"])[^'\"\r\n]+\1", "", answer_without_inline_commands
    )
    normalized_answer = unicodedata.normalize("NFKC", answer_without_inline_commands)
    for match in re.finditer(re.escape(expected), normalized_answer):
        prefix = normalized_answer[: match.start()]
        prefix_without_spaces = prefix.rstrip(" \t")
        if prefix_without_spaces:
            last_line = prefix_without_spaces.rsplit("\n", maxsplit=1)[-1].strip()
            previous_word = last_line.rsplit(maxsplit=1)[-1].lower() if last_line else ""
            starts_command_line = prefix_without_spaces.endswith(("\r", "\n")) or last_line in {
                "-",
                "*",
            }
            if (
                not starts_command_line
                and previous_word not in {"run", "execute", "use", "is", "command"}
                and not last_line.endswith(":")
            ):
                continue
        suffix = normalized_answer[match.end() :]
        if not suffix:
            return True
        suffix_without_spaces = suffix.lstrip(" \t")
        if not suffix_without_spaces or suffix_without_spaces[0] in "`.,;:!?)]}\r\n":
            return True
    return False


def _is_customer_name_present(expected: str, answer: str) -> bool:
    """Match a required name while tolerating harmless spacing and punctuation variation."""

    expected_parts = re.findall(r"[a-z0-9]+", unicodedata.normalize("NFKC", expected).casefold())
    if not expected_parts:
        return False
    normalized_answer = unicodedata.normalize("NFKC", answer).casefold()
    flexible_name = r"[^a-z0-9]*".join(re.escape(part) for part in expected_parts)
    return re.search(rf"(?<![a-z0-9]){flexible_name}(?![a-z0-9])", normalized_answer) is not None


def _is_present(
    expected: str,
    *,
    answer: str,
    normalized_answer: str,
    normalized_question: str,
    match_kind: _ContentMatchKind,
) -> bool:
    if match_kind == "date":
        return _is_date_present(expected, normalized_answer, normalized_question)
    if match_kind == "command":
        return _is_exact_command_present(expected, answer)
    if match_kind == "customer":
        return _is_customer_name_present(expected, answer)
    return _contains_normalized_fragment(normalized_answer, _normalize(expected))


def _coverage(matched: int, total: int) -> float | None:
    return matched / total if total else None


def _content_check(
    *,
    name: str,
    expected: Sequence[str],
    answer: str,
    normalized_answer: str,
    normalized_question: str,
    is_gate: bool,
    match_kind: _ContentMatchKind,
) -> tuple[CheckResult, HitCount | None]:
    missing = [
        value
        for value in expected
        if not _is_present(
            value,
            answer=answer,
            normalized_answer=normalized_answer,
            normalized_question=normalized_question,
            match_kind=match_kind,
        )
    ]
    matched = len(expected) - len(missing)
    return (
        CheckResult(
            name=name,
            scope="content",
            passed=not missing,
            score=_coverage(matched, len(expected)),
            is_gate=is_gate,
            details=f"missing: {missing}" if missing else "",
        ),
        HitCount(matched=matched, total=len(expected)) if expected else None,
    )


def _diagnostic_source_check(
    *,
    name: str,
    diagnostic_source_ids: Sequence[str],
    observed_source_ids: Sequence[str],
) -> CheckResult:
    expected = set(diagnostic_source_ids)
    observed = set(observed_source_ids)
    missing = sorted(expected - observed)
    matched = len(expected) - len(missing)
    return CheckResult(
        name=name,
        scope="diagnostic",
        passed=not missing,
        score=_coverage(matched, len(expected)),
        details=f"missing candidate-curated IDs: {missing}" if missing else "",
    )


def _gated_scope_passed(checks: Sequence[CheckResult], scope: CheckScope) -> bool:
    return all(check.passed for check in checks if check.is_gate and check.scope == scope)


def _optional_gated_scope_passed(checks: Sequence[CheckResult], scope: CheckScope) -> bool | None:
    applicable_checks = [check for check in checks if check.is_gate and check.scope == scope]
    return all(check.passed for check in applicable_checks) if applicable_checks else None


def evaluate_response(
    case: EvalCase,
    response: AgentResponse,
    *,
    duration_ms: int | None = None,
) -> EvalResult:
    normalized_answer = _normalize(response.answer)
    normalized_question = _normalize(case.question)
    checks: list[CheckResult] = []
    lexical_hits: dict[str, HitCount] = {}
    content_groups: tuple[tuple[str, str, Sequence[str], bool, _ContentMatchKind], ...] = (
        ("facts", "lexical_fact_anchors", case.expected_facts, False, "lexical"),
        ("entities", "lexical_entities", case.expected_entities, False, "lexical"),
        ("dates", "exact_dates", case.expected_dates, True, "date"),
        ("commands", "exact_commands", case.expected_commands, True, "command"),
        (
            "customers",
            "required_customer_recall",
            case.expected_customers,
            True,
            "customer",
        ),
    )
    for group, check_name, expected, is_gate, match_kind in content_groups:
        if not expected:
            continue
        check, hit_count = _content_check(
            name=check_name,
            expected=expected,
            answer=response.answer,
            normalized_answer=normalized_answer,
            normalized_question=normalized_question,
            is_gate=is_gate,
            match_kind=match_kind,
        )
        checks.append(check)
        if hit_count is not None:
            lexical_hits[group] = hit_count

    source_ids = [source.artifact_id for source in response.sources]
    unknown_citations = sorted(set(source_ids) - set(response.retrieved_artifact_ids))
    forbidden_found = [
        value for value in case.forbidden_phrases if _normalize(value) in normalized_answer
    ]
    checks.extend(
        [
            CheckResult(
                name="source_attribution",
                scope="evidence",
                passed=bool(source_ids) or case.insufficient_evidence_acceptable,
                is_gate=True,
                details="no sources returned" if not source_ids else "",
            ),
            CheckResult(
                name="citation_integrity",
                scope="evidence",
                passed=not unknown_citations,
                is_gate=True,
                details=(
                    f"citations absent from retrieved evidence: {unknown_citations}"
                    if unknown_citations
                    else ""
                ),
            ),
            _diagnostic_source_check(
                name="diagnostic_citation_coverage",
                diagnostic_source_ids=case.diagnostic_source_ids,
                observed_source_ids=source_ids,
            ),
            _diagnostic_source_check(
                name="diagnostic_retrieval_coverage",
                diagnostic_source_ids=case.diagnostic_source_ids,
                observed_source_ids=response.retrieved_artifact_ids,
            ),
            CheckResult(
                name="tool_call_budget",
                scope="operations",
                passed=response.tool_call_count <= case.max_tool_calls,
                is_gate=True,
                details=(
                    f"{response.tool_call_count} > {case.max_tool_calls}"
                    if response.tool_call_count > case.max_tool_calls
                    else ""
                ),
            ),
            CheckResult(
                name="retrieval_round_budget",
                scope="operations",
                passed=response.retrieval_round_count <= case.max_retrieval_rounds,
                is_gate=True,
                details=(
                    f"{response.retrieval_round_count} > {case.max_retrieval_rounds}"
                    if response.retrieval_round_count > case.max_retrieval_rounds
                    else ""
                ),
            ),
            CheckResult(
                name="answerability_behavior",
                scope="evidence",
                passed=(
                    response.insufficient_evidence is case.expected_insufficient_evidence
                    if case.expected_insufficient_evidence is not None
                    else not response.insufficient_evidence
                ),
                is_gate=True,
                details=(
                    "expected insufficient-evidence response, but the agent claimed an answer"
                    if case.expected_insufficient_evidence is True
                    and not response.insufficient_evidence
                    else (
                        "agent returned an insufficient-evidence response for an answerable case"
                        if response.insufficient_evidence
                        and case.expected_insufficient_evidence is not True
                        else ""
                    )
                ),
            ),
        ]
    )
    if case.expected_show_sources is not None:
        checks.append(
            CheckResult(
                name="source_visibility",
                scope="evidence",
                passed=response.show_sources is case.expected_show_sources,
                is_gate=True,
                details=(
                    f"expected {case.expected_show_sources}, got {response.show_sources}"
                    if response.show_sources is not case.expected_show_sources
                    else ""
                ),
            )
        )
    if case.forbidden_phrases:
        checks.append(
            CheckResult(
                name="forbidden_phrases",
                scope="safety",
                passed=not forbidden_found,
                is_gate=True,
                details=f"found: {forbidden_found}" if forbidden_found else "",
            )
        )
    if case.max_model_calls is not None:
        checks.append(
            CheckResult(
                name="model_call_budget",
                scope="operations",
                passed=response.model_call_count <= case.max_model_calls,
                is_gate=True,
                details=(
                    f"{response.model_call_count} > {case.max_model_calls}"
                    if response.model_call_count > case.max_model_calls
                    else ""
                ),
            )
        )
    strict_contract_passed = all(check.passed for check in checks if check.is_gate)
    return EvalResult(
        case_id=case.id,
        strict_contract_passed=strict_contract_passed,
        content_exact_passed=_optional_gated_scope_passed(checks, "content"),
        evidence_passed=_gated_scope_passed(checks, "evidence"),
        operational_passed=_gated_scope_passed(checks, "operations"),
        safety_passed=_optional_gated_scope_passed(checks, "safety"),
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
        lexical_hits=lexical_hits,
    )
