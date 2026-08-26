"""Schema-aware, read-only access to the supplied SQLite knowledge database."""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from slack_qa_agent.retrieval.database import open_read_only_database
from slack_qa_agent.retrieval.models import (
    AccountLookupInput,
    EvidenceItem,
    ReadArtifactsInput,
    SearchHit,
    SearchKnowledgeInput,
)

MAX_STRUCTURED_EVIDENCE_CHARS = 1_500

_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ID_COLUMNS = ("artifact_id", "id", "uuid")
_TITLE_COLUMNS = ("title", "name", "subject")
_CONTENT_COLUMNS = ("content", "content_text", "body", "text", "raw_text", "document")
_SUMMARY_COLUMNS = ("summary", "snippet", "description")
_TYPE_COLUMNS = ("artifact_type", "type", "kind")
_METADATA_COLUMNS = ("metadata", "metadata_json")
_CUSTOMER_COLUMNS = ("customer", "customer_name", "account", "account_name")


class UnsupportedKnowledgeSchema(RuntimeError):
    pass


@dataclass(frozen=True)
class KnowledgeSchema:
    artifact_table: str
    id_column: str
    title_column: str
    content_column: str
    summary_column: str | None
    type_column: str | None
    metadata_column: str | None
    customer_column: str | None
    fts_table: str | None
    fts_id_column: str | None


def _quote(identifier: str) -> str:
    if not _IDENTIFIER.fullmatch(identifier):
        raise UnsupportedKnowledgeSchema(f"Unsafe SQLite identifier: {identifier!r}")
    return f'"{identifier}"'


def _first(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _fts_query(value: str) -> str:
    """Convert user/model text to literal FTS tokens, excluding FTS operators."""

    tokens = re.findall(r"[A-Za-z0-9_./:-]+", value)[:32]
    return " OR ".join(f'"{token}"' for token in tokens)


def discover_schema(connection: sqlite3.Connection) -> KnowledgeSchema:
    objects = connection.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    if not objects:
        raise UnsupportedKnowledgeSchema("Knowledge database has no application tables")

    fts_tables = {
        str(row["name"]): str(row["sql"] or "")
        for row in objects
        if "using fts5" in str(row["sql"] or "").lower()
    }
    candidates: list[tuple[int, KnowledgeSchema]] = []
    for row in objects:
        table = str(row["name"])
        if table in fts_tables or table.startswith("fts_") or table.endswith(("_data", "_idx")):
            continue
        columns = {
            str(column["name"]).lower()
            for column in connection.execute(f"PRAGMA table_info({_quote(table)})").fetchall()
        }
        id_column = _first(columns, _ID_COLUMNS)
        title_column = _first(columns, _TITLE_COLUMNS)
        content_column = _first(columns, _CONTENT_COLUMNS)
        if not (id_column and title_column and content_column):
            continue

        matching_fts = next(
            (
                fts_name
                for fts_name, ddl in fts_tables.items()
                if re.search(rf"content\s*=\s*['\"]{re.escape(table)}['\"]", ddl, re.IGNORECASE)
                or fts_name in {f"{table}_fts", f"{table}_fts5", "artifacts_fts"}
            ),
            None,
        )
        fts_columns = (
            {
                str(column["name"]).lower()
                for column in connection.execute(
                    f"PRAGMA table_info({_quote(matching_fts)})"
                ).fetchall()
            }
            if matching_fts
            else set()
        )
        score = (100 if table == "artifacts" else 0) + (25 if matching_fts else 0)
        candidates.append(
            (
                score,
                KnowledgeSchema(
                    artifact_table=table,
                    id_column=id_column,
                    title_column=title_column,
                    content_column=content_column,
                    summary_column=_first(columns, _SUMMARY_COLUMNS),
                    type_column=_first(columns, _TYPE_COLUMNS),
                    metadata_column=_first(columns, _METADATA_COLUMNS),
                    customer_column=_first(columns, _CUSTOMER_COLUMNS),
                    fts_table=matching_fts,
                    fts_id_column=id_column if id_column in fts_columns else None,
                ),
            )
        )

    if not candidates:
        described = ", ".join(str(row["name"]) for row in objects)
        raise UnsupportedKnowledgeSchema(
            "Could not identify an artifact table with ID, title, and content columns. "
            f"Found: {described}"
        )
    return max(candidates, key=lambda item: item[0])[1]


class SQLiteKnowledgeRepository:
    def __init__(self, path: Path) -> None:
        self._path = path

    def inspect_schema(self) -> KnowledgeSchema:
        with open_read_only_database(self._path) as connection:
            return discover_schema(connection)

    def inspect_assignment_schema(self) -> KnowledgeSchema:
        """Validate both artifact search and the assignment's relational account schema."""

        with open_read_only_database(self._path) as connection:
            schema = discover_schema(connection)
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
            schema = discover_schema(connection)
            rows = self._search_rows(connection, schema, request)
        return [self._to_search_hit(row, schema) for row in rows]

    def read(self, request: ReadArtifactsInput) -> list[EvidenceItem]:
        with open_read_only_database(self._path) as connection:
            schema = discover_schema(connection)
            ids = request.artifact_ids
            placeholders = ",".join("?" for _ in ids)
            rows = connection.execute(
                f"SELECT * FROM {_quote(schema.artifact_table)} "
                f"WHERE {_quote(schema.id_column)} IN ({placeholders})",
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

        clauses: list[str] = []
        parameters: list[Any] = []
        for column, value in (
            ("c.region", request.region),
            ("c.country", request.country),
            ("p.name", request.product),
        ):
            if value is not None:
                clauses.append(f"{column} = ?")
                parameters.append(value)
        if request.pain_point_terms:
            term_clauses = []
            for term in request.pain_point_terms:
                term_clauses.append("lower(s.pain_point) LIKE ? ESCAPE '\\'")
                escaped = term.lower().replace("\\", "\\\\").replace("%", "\\%")
                escaped = escaped.replace("_", "\\_")
                parameters.append(f"%{escaped}%")
            clauses.append(f"({' OR '.join(term_clauses)})")

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
            + " AND ".join(clauses)
            + """
            ORDER BY
                c.name,
                CASE a.artifact_type WHEN 'support_ticket' THEN 0 ELSE 1 END,
                a.created_at,
                a.artifact_id
        """
        )
        with open_read_only_database(self._path) as connection:
            schema = discover_schema(connection)
            rows = connection.execute(query, parameters).fetchall()

        selected: list[EvidenceItem] = []
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
            selected.append(EvidenceItem(**hit.model_dump(), content=content))
            if len(selected) >= request.limit:
                break
        return selected

    def _search_rows(
        self,
        connection: sqlite3.Connection,
        schema: KnowledgeSchema,
        request: SearchKnowledgeInput,
    ) -> list[sqlite3.Row]:
        table = _quote(schema.artifact_table)
        where: list[str] = []
        parameters: list[Any] = []
        if request.filters.artifact_type:
            if schema.type_column is None:
                return []
            where.append(f"a.{_quote(schema.type_column)} = ?")
            parameters.append(request.filters.artifact_type)
        if request.filters.customer:
            if schema.customer_column is None:
                return []
            where.append(f"a.{_quote(schema.customer_column)} = ?")
            parameters.append(request.filters.customer)

        if schema.fts_table:
            fts_query = _fts_query(request.query)
            if not fts_query:
                return []
            fts = _quote(schema.fts_table)
            join_condition = (
                f"a.{_quote(schema.id_column)} = {fts}.{_quote(schema.fts_id_column)}"
                if schema.fts_id_column
                else f"a.rowid = {fts}.rowid"
            )
            clauses = [f"{fts} MATCH ?", *where]
            query = (
                f"SELECT a.*, bm25({fts}) AS _retrieval_score FROM {fts} "
                f"JOIN {table} AS a ON {join_condition} "
                f"WHERE {' AND '.join(clauses)} ORDER BY _retrieval_score LIMIT ?"
            )
            return connection.execute(query, [fts_query, *parameters, request.limit]).fetchall()

        terms = [term for term in re.findall(r"[A-Za-z0-9_-]+", request.query) if len(term) > 1]
        if not terms:
            return []
        lexical: list[str] = []
        lexical_parameters: list[str] = []
        for term in terms[:8]:
            lexical.append(
                f"(a.{_quote(schema.title_column)} LIKE ? ESCAPE '\\' "
                f"OR a.{_quote(schema.content_column)} LIKE ? ESCAPE '\\')"
            )
            escaped = term.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            lexical_parameters.extend([f"%{escaped}%", f"%{escaped}%"])
        clauses = [f"({' OR '.join(lexical)})", *where]
        query = f"SELECT a.*, NULL AS _retrieval_score FROM {table} AS a WHERE "
        query += " AND ".join(clauses) + f" ORDER BY a.{_quote(schema.title_column)} LIMIT ?"
        return connection.execute(
            query, [*lexical_parameters, *parameters, request.limit]
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
