"""Deterministic parsing and validation of artifact citations."""

from __future__ import annotations

import re

from knowledge_assistant.retrieval.models import EvidenceItem

_CITATION_PATTERN = re.compile(r"\[([^\]\s]+)\]")


def cited_artifact_ids(answer: str) -> set[str]:
    return set(_CITATION_PATTERN.findall(answer))


def citation_issues(answer: str, evidence: list[EvidenceItem]) -> list[str]:
    cited_ids = cited_artifact_ids(answer)
    if not cited_ids:
        return ["Answer does not cite any retrieved artifact."]

    evidence_ids = {item.artifact_id for item in evidence}
    unknown_ids = sorted(cited_ids - evidence_ids)
    if unknown_ids:
        return [f"Answer cites artifacts that were not retrieved: {', '.join(unknown_ids)}"]
    return []
