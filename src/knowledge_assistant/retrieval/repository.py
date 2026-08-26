"""Strict, read-only access to the supplied SQLite knowledge database."""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from typing import Any

from knowledge_assistant.retrieval.database import open_read_only_database
from knowledge_assistant.retrieval.models import (
    AccountLookupInput,
    EvidenceItem,
    ReadArtifactsInput,
    SearchHit,
    SearchKnowledgeInput,
)
from knowledge_assistant.retrieval.schema import KnowledgeSchema, validate_knowledge_schema

MAX_STRUCTURED_EVIDENCE_CHARS = 1_500


def _escape_like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fts_query(value: str) -> str:
    """Convert user/model text to literal FTS tokens, excluding FTS operators."""

    tokens = re.findall(r"[A-Za-z0-9_./:-]+", value)[:32]
    # OR is recall-first; the bounded grading stage decides relevance.
    return " OR ".join(f'"{token}"' for token in tokens)


def _build_account_lookup_query(request: AccountLookupInput) -> tuple[str, list[Any]]:
    filter_clauses: list[str] = []
    query_parameters: list[Any] = []
    for column, value in (
        ("c.region", request.region),
        ("c.country", request.country),
        ("p.name", request.product),
    ):
        if value is not None:
            filter_clauses.append(f"{column} = ?")
            query_parameters.append(value)

    if request.pain_point_terms:
        pain_point_clauses = []
        for term in request.pain_point_terms:
            pain_point_clauses.append("lower(s.pain_point) LIKE ? ESCAPE '\\'")
            query_parameters.append(f"%{_escape_like_pattern(term.lower())}%")
        filter_clauses.append(f"({' OR '.join(pain_point_clauses)})")

    query = (
        """
        SELECT
            c.name AS _customer_name,
            c.region AS _customer_region,
            c.country AS _customer_country,
            p.name AS _product_name,
            s.pain_point AS _pain_point,
            s.trigger_event AS _trigger_event,
            a.*,
            NULL AS _retrieval_score
        FROM customers AS c
        JOIN implementations AS i ON i.customer_id = c.customer_id
        JOIN products AS p ON p.product_id = i.product_id
        JOIN scenarios AS s ON s.scenario_id = c.scenario_id
        JOIN artifacts AS a ON a.scenario_id = s.scenario_id
        WHERE """
        + " AND ".join(filter_clauses)
        + """
        ORDER BY
            c.name,
            CASE a.artifact_type WHEN 'support_ticket' THEN 0 ELSE 1 END,
            a.created_at,
            a.artifact_id
        """
    )
    return query, query_parameters


class SQLiteKnowledgeRepository:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._validated_schema: KnowledgeSchema | None = None

    def inspect_schema(self) -> KnowledgeSchema:
        return self.validate_runtime_schema()

    def validate_runtime_schema(self) -> KnowledgeSchema:
        with open_read_only_database(self._path) as connection:
            return self._require_schema(connection)

    def search(self, request: SearchKnowledgeInput) -> list[SearchHit]:
        with open_read_only_database(self._path) as connection:
            schema = self._require_schema(connection)
            rows = self._search_fts_rows(connection, request)
        return [self._to_search_hit(row, schema) for row in rows]

    def read(self, request: ReadArtifactsInput) -> list[EvidenceItem]:
        with open_read_only_database(self._path) as connection:
            schema = self._require_schema(connection)
            placeholders = ",".join("?" for _ in request.artifact_ids)
            rows = connection.execute(
                f"SELECT * FROM artifacts WHERE artifact_id IN ({placeholders})",
                request.artifact_ids,
            ).fetchall()

        by_id = {str(row["artifact_id"]): row for row in rows}
        remaining = request.max_context_chars
        evidence: list[EvidenceItem] = []
        for artifact_id in request.artifact_ids:
            row = by_id.get(artifact_id)
            if row is None or remaining <= 0:
                continue
            content = str(row["content_text"] or "")[:remaining]
            remaining -= len(content)
            hit = self._to_search_hit(row, schema)
            evidence.append(EvidenceItem(**hit.model_dump(), content=content))
        return evidence

    def lookup_accounts(self, request: AccountLookupInput) -> list[EvidenceItem]:
        """Return one representative artifact per account using parameterized joins."""

        query, query_parameters = _build_account_lookup_query(request)
        with open_read_only_database(self._path) as connection:
            schema = self._require_schema(connection)
            rows = connection.execute(query, query_parameters).fetchall()

        representative_evidence: list[EvidenceItem] = []
        seen_customers: set[str] = set()
        for row in rows:
            customer = str(row["_customer_name"])
            if customer in seen_customers:
                continue
            seen_customers.add(customer)
            content = json.dumps(
                {
                    "customer": customer,
                    "region": row["_customer_region"],
                    "country": row["_customer_country"],
                    "product": row["_product_name"],
                    "pain_point": row["_pain_point"],
                    "trigger_event": row["_trigger_event"],
                    "artifact_excerpt": str(row["content_text"] or ""),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )[:MAX_STRUCTURED_EVIDENCE_CHARS]
            hit = self._to_search_hit(row, schema)
            representative_evidence.append(EvidenceItem(**hit.model_dump(), content=content))
            if len(representative_evidence) >= request.limit:
                break
        return representative_evidence

    def _require_schema(self, connection: sqlite3.Connection) -> KnowledgeSchema:
        if self._validated_schema is None:
            self._validated_schema = validate_knowledge_schema(connection)
        return self._validated_schema

    @staticmethod
    def _search_fts_rows(
        connection: sqlite3.Connection,
        request: SearchKnowledgeInput,
    ) -> list[sqlite3.Row]:
        fts_query = _fts_query(request.query)
        if not fts_query:
            raise ValueError("search query does not contain any FTS-compatible tokens")

        filter_clauses: list[str] = []
        filter_parameters: list[Any] = []
        if request.filters.artifact_type:
            filter_clauses.append("a.artifact_type = ?")
            filter_parameters.append(request.filters.artifact_type)

        where_clauses = ["artifacts_fts MATCH ?", *filter_clauses]
        query = (
            "SELECT a.*, bm25(artifacts_fts) AS _retrieval_score FROM artifacts_fts "
            "JOIN artifacts AS a ON a.artifact_id = artifacts_fts.artifact_id "
            f"WHERE {' AND '.join(where_clauses)} ORDER BY _retrieval_score LIMIT ?"
        )
        return connection.execute(
            query,
            [fts_query, *filter_parameters, request.limit],
        ).fetchall()

    @staticmethod
    def _to_search_hit(row: sqlite3.Row, schema: KnowledgeSchema) -> SearchHit:
        content = str(row["content_text"] or "")
        raw_metadata = row["metadata_json"]
        metadata: dict[str, Any] = {}
        if raw_metadata:
            try:
                parsed = json.loads(str(raw_metadata))
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid metadata_json for artifact {row['artifact_id']}"
                ) from exc
            if not isinstance(parsed, dict):
                raise ValueError(
                    f"metadata_json must be an object for artifact {row['artifact_id']}"
                )
            metadata = parsed

        keys = set(row.keys())
        raw_score = row["_retrieval_score"] if "_retrieval_score" in keys else None
        return SearchHit(
            artifact_id=str(row[schema.id_column]),
            title=str(row[schema.title_column]),
            artifact_type=str(row[schema.type_column]) if row[schema.type_column] else None,
            snippet=(str(row[schema.summary_column] or "") or content)[:500],
            score=float(raw_score) if raw_score is not None else None,
            metadata=metadata,
        )
