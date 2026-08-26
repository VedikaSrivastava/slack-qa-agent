"""Schema-aware, read-only access to the supplied SQLite knowledge database."""

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
from knowledge_assistant.retrieval.schema import (
    KnowledgeSchema,
    discover_knowledge_schema,
    quote_identifier,
)

MAX_STRUCTURED_EVIDENCE_CHARS = 1_500


def _escape_like_pattern(value: str) -> str:
    return value.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")


def _fts_query(value: str) -> str:
    """Convert user/model text to literal FTS tokens, excluding FTS operators."""

    tokens = re.findall(r"[A-Za-z0-9_./:-]+", value)[:32]
    # OR is intentionally recall-first; the bounded grading stage decides relevance.
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

    def inspect_schema(self) -> KnowledgeSchema:
        with open_read_only_database(self._path) as connection:
            return discover_knowledge_schema(connection)

    def validate_runtime_schema(self) -> KnowledgeSchema:
        """Validate every table and relationship required by runtime retrieval."""

        with open_read_only_database(self._path) as connection:
            schema = discover_knowledge_schema(connection)
            connection.execute(
                """
                SELECT c.name, c.region, c.country, p.name, s.pain_point, s.trigger_event
                FROM customers AS c
                JOIN implementations AS i ON i.customer_id = c.customer_id
                JOIN products AS p ON p.product_id = i.product_id
                JOIN scenarios AS s ON s.scenario_id = c.scenario_id
                LIMIT 0
                """
            )
            return schema

    def search(self, request: SearchKnowledgeInput) -> list[SearchHit]:
        with open_read_only_database(self._path) as connection:
            schema = discover_knowledge_schema(connection)
            rows = self._search_rows(connection, schema, request)
        return [self._to_search_hit(row, schema) for row in rows]

    def read(self, request: ReadArtifactsInput) -> list[EvidenceItem]:
        with open_read_only_database(self._path) as connection:
            schema = discover_knowledge_schema(connection)
            ids = request.artifact_ids
            placeholders = ",".join("?" for _ in ids)
            rows = connection.execute(
                f"SELECT * FROM {quote_identifier(schema.artifact_table)} "
                f"WHERE {quote_identifier(schema.id_column)} IN ({placeholders})",
                ids,
            ).fetchall()

        by_id = {str(row[schema.id_column]): row for row in rows}
        remaining = request.max_context_chars
        evidence: list[EvidenceItem] = []
        for artifact_id in ids:
            row = by_id.get(artifact_id)
            if row is None or remaining <= 0:
                continue
            content = str(row[schema.content_column] or "")[:remaining]
            remaining -= len(content)
            hit = self._to_search_hit(row, schema)
            evidence.append(EvidenceItem(**hit.model_dump(), content=content))
        return evidence

    def lookup_accounts(self, request: AccountLookupInput) -> list[EvidenceItem]:
        """Return one representative artifact per account using fixed, parameterized joins."""

        query, query_parameters = _build_account_lookup_query(request)
        with open_read_only_database(self._path) as connection:
            schema = discover_knowledge_schema(connection)
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
                    "artifact_excerpt": str(row[schema.content_column] or ""),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )[:MAX_STRUCTURED_EVIDENCE_CHARS]
            hit = self._to_search_hit(row, schema)
            # One artifact per account keeps cross-account context bounded and balanced.
            representative_evidence.append(EvidenceItem(**hit.model_dump(), content=content))
            if len(representative_evidence) >= request.limit:
                break
        return representative_evidence

    def _search_rows(
        self,
        connection: sqlite3.Connection,
        schema: KnowledgeSchema,
        request: SearchKnowledgeInput,
    ) -> list[sqlite3.Row]:
        table = quote_identifier(schema.artifact_table)
        filter_clauses: list[str] = []
        filter_parameters: list[Any] = []
        if request.filters.artifact_type:
            if schema.type_column is None:
                return []
            filter_clauses.append(f"a.{quote_identifier(schema.type_column)} = ?")
            filter_parameters.append(request.filters.artifact_type)
        if request.filters.customer:
            if schema.customer_column is None:
                return []
            filter_clauses.append(f"a.{quote_identifier(schema.customer_column)} = ?")
            filter_parameters.append(request.filters.customer)

        if schema.fts_table:
            return self._search_fts_rows(
                connection, schema, request, table, filter_clauses, filter_parameters
            )
        return self._search_lexical_rows(
            connection, schema, request, table, filter_clauses, filter_parameters
        )

    @staticmethod
    def _search_fts_rows(
        connection: sqlite3.Connection,
        schema: KnowledgeSchema,
        request: SearchKnowledgeInput,
        artifact_table: str,
        filter_clauses: list[str],
        filter_parameters: list[Any],
    ) -> list[sqlite3.Row]:
        fts_query = _fts_query(request.query)
        if not fts_query or schema.fts_table is None:
            return []
        fts_table = quote_identifier(schema.fts_table)
        join_condition = (
            f"a.{quote_identifier(schema.id_column)} = "
            f"{fts_table}.{quote_identifier(schema.fts_id_column)}"
            if schema.fts_id_column
            else f"a.rowid = {fts_table}.rowid"
        )
        where_clauses = [f"{fts_table} MATCH ?", *filter_clauses]
        query = (
            f"SELECT a.*, bm25({fts_table}) AS _retrieval_score FROM {fts_table} "
            f"JOIN {artifact_table} AS a ON {join_condition} "
            f"WHERE {' AND '.join(where_clauses)} ORDER BY _retrieval_score LIMIT ?"
        )
        parameters = [fts_query, *filter_parameters, request.limit]
        return connection.execute(query, parameters).fetchall()

    @staticmethod
    def _search_lexical_rows(
        connection: sqlite3.Connection,
        schema: KnowledgeSchema,
        request: SearchKnowledgeInput,
        artifact_table: str,
        filter_clauses: list[str],
        filter_parameters: list[Any],
    ) -> list[sqlite3.Row]:

        terms = [term for term in re.findall(r"[A-Za-z0-9_-]+", request.query) if len(term) > 1]
        if not terms:
            return []
        lexical_clauses: list[str] = []
        lexical_parameters: list[str] = []
        for term in terms[:8]:
            lexical_clauses.append(
                f"(a.{quote_identifier(schema.title_column)} LIKE ? ESCAPE '\\' "
                f"OR a.{quote_identifier(schema.content_column)} LIKE ? ESCAPE '\\')"
            )
            escaped = _escape_like_pattern(term)
            lexical_parameters.extend([f"%{escaped}%", f"%{escaped}%"])
        where_clauses = [f"({' OR '.join(lexical_clauses)})", *filter_clauses]
        query = f"SELECT a.*, NULL AS _retrieval_score FROM {artifact_table} AS a WHERE "
        query += (
            " AND ".join(where_clauses)
            + f" ORDER BY a.{quote_identifier(schema.title_column)} LIMIT ?"
        )
        return connection.execute(
            query, [*lexical_parameters, *filter_parameters, request.limit]
        ).fetchall()

    @staticmethod
    def _to_search_hit(row: sqlite3.Row, schema: KnowledgeSchema) -> SearchHit:
        content = str(row[schema.content_column] or "")
        summary = str(row[schema.summary_column] or "") if schema.summary_column else ""
        raw_metadata = row[schema.metadata_column] if schema.metadata_column else None
        metadata: dict[str, Any] = {}
        if raw_metadata:
            try:
                parsed = json.loads(str(raw_metadata))
                if isinstance(parsed, dict):
                    metadata = parsed
            except json.JSONDecodeError:
                metadata = {"raw": str(raw_metadata)[:1_000]}
        keys = set(row.keys())
        raw_score = row["_retrieval_score"] if "_retrieval_score" in keys else None
        return SearchHit(
            artifact_id=str(row[schema.id_column]),
            title=str(row[schema.title_column]),
            artifact_type=str(row[schema.type_column])
            if schema.type_column and row[schema.type_column]
            else None,
            snippet=(summary or content)[:500],
            score=float(raw_score) if raw_score is not None else None,
            metadata=metadata,
        )
