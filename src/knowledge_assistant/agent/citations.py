"""Deterministic parsing and validation of artifact citations."""

from __future__ import annotations

import re

from knowledge_assistant.agent.models import EvidenceReference
from knowledge_assistant.retrieval.models import EvidenceItem

# Supplied artifact IDs use the `art_` prefix. Search only inside square brackets so ordinary
# prose and Markdown labels are ignored, while accepting both `[art_a][art_b]` and the grouped
# citation style models commonly produce: `[art_a, art_b]`.
_BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_ARTIFACT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])(art_[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])")

# Labels naming the scaffolding blocks supplied to the model. Answers occasionally echo one as if
# it were a citable source, which exposes retrieval internals to Slack readers.
_INTERNAL_BLOCK_LABELS = frozenset(
    {"ACCOUNT_LOOKUP_COVERAGE", "PLANNED_COMPARISON_FOLLOW_UP_QUERIES"}
)


def cited_artifact_ids(answer: str) -> set[str]:
    return {
        artifact_id
        for bracket_content in _BRACKET_PATTERN.findall(answer)
        for artifact_id in _ARTIFACT_ID_PATTERN.findall(bracket_content)
    }


def hide_internal_markers(answer: str) -> str:
    """Remove provenance markers and prompt-block labels, leaving ordinary bracketed prose intact."""

    def replace_marker(match: re.Match[str]) -> str:
        bracket_content = match.group(1)
        if _ARTIFACT_ID_PATTERN.search(bracket_content):
            return ""
        if bracket_content.strip() in _INTERNAL_BLOCK_LABELS:
            return ""
        return match.group(0)

    return re.sub(r"[ \t]*\[([^\[\]]+)\]", replace_marker, answer)


def citation_issues(answer: str, evidence: list[EvidenceItem]) -> list[str]:
    cited_ids = cited_artifact_ids(answer)
    if not cited_ids:
        return ["Answer does not cite any retrieved artifact."]

    evidence_ids = {item.artifact_id for item in evidence}
    unknown_ids = sorted(cited_ids - evidence_ids)
    if unknown_ids:
        return [f"Answer cites artifacts that were not retrieved: {', '.join(unknown_ids)}"]
    return []


def references_for_cited_evidence(
    answer: str,
    evidence: list[EvidenceItem],
) -> list[EvidenceReference]:
    """Build compact ordered provenance without copying retrieved text into thread history."""

    cited_ids = cited_artifact_ids(answer)
    return [
        EvidenceReference(
            artifact_id=item.artifact_id,
            title=item.title,
            score=item.score,
        )
        for item in evidence
        if item.artifact_id in cited_ids
    ]
