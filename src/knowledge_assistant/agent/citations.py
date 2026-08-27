"""Deterministic parsing and validation of artifact citations."""

from __future__ import annotations

import re

from knowledge_assistant.retrieval.models import EvidenceItem

# Supplied artifact IDs use the `art_` prefix. Search only inside square brackets so ordinary
# prose and Markdown labels are ignored, while accepting both `[art_a][art_b]` and the grouped
# citation style models commonly produce: `[art_a, art_b]`.
_BRACKET_PATTERN = re.compile(r"\[([^\[\]]+)\]")
_ARTIFACT_ID_PATTERN = re.compile(r"(?<![A-Za-z0-9_-])(art_[A-Za-z0-9_-]+)(?![A-Za-z0-9_-])")


def cited_artifact_ids(answer: str) -> set[str]:
    return {
        artifact_id
        for bracket_content in _BRACKET_PATTERN.findall(answer)
        for artifact_id in _ARTIFACT_ID_PATTERN.findall(bracket_content)
    }


def citation_issues(answer: str, evidence: list[EvidenceItem]) -> list[str]:
    cited_ids = cited_artifact_ids(answer)
    if not cited_ids:
        return ["Answer does not cite any retrieved artifact."]

    evidence_ids = {item.artifact_id for item in evidence}
    unknown_ids = sorted(cited_ids - evidence_ids)
    if unknown_ids:
        return [f"Answer cites artifacts that were not retrieved: {', '.join(unknown_ids)}"]
    return []
