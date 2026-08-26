"""Discovery and validation of the supplied knowledge-database schema."""

from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass

_SAFE_IDENTIFIER = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_ID_COLUMNS = ("artifact_id", "id", "uuid")
_TITLE_COLUMNS = ("title", "name", "subject")
_CONTENT_COLUMNS = ("content", "content_text", "body", "text", "raw_text", "document")
_SUMMARY_COLUMNS = ("summary", "snippet", "description")
_TYPE_COLUMNS = ("artifact_type", "type", "kind")
_METADATA_COLUMNS = ("metadata", "metadata_json")
_CUSTOMER_COLUMNS = ("customer", "customer_name", "account", "account_name")


class UnsupportedKnowledgeSchema(RuntimeError):
    """Raised when the supplied database cannot support the retrieval contract."""


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


def quote_identifier(identifier: str) -> str:
    """Quote a discovered SQLite identifier after enforcing a strict allowlist."""

    if not _SAFE_IDENTIFIER.fullmatch(identifier):
        raise UnsupportedKnowledgeSchema(f"Unsafe SQLite identifier: {identifier!r}")
    return f'"{identifier}"'


def _first_matching_column(columns: set[str], candidates: tuple[str, ...]) -> str | None:
    return next((candidate for candidate in candidates if candidate in columns), None)


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    return {
        str(column["name"]).lower()
        for column in connection.execute(
            f"PRAGMA table_info({quote_identifier(table_name)})"
        ).fetchall()
    }


def _matching_fts_table(
    artifact_table: str,
    fts_tables: dict[str, str],
) -> str | None:
    expected_names = {f"{artifact_table}_fts", f"{artifact_table}_fts5", "artifacts_fts"}
    return next(
        (
            table_name
            for table_name, definition in fts_tables.items()
            if re.search(
                rf"content\s*=\s*['\"]{re.escape(artifact_table)}['\"]",
                definition,
                re.IGNORECASE,
            )
            or table_name in expected_names
        ),
        None,
    )


def discover_knowledge_schema(connection: sqlite3.Connection) -> KnowledgeSchema:
    """Select the best artifact table and its optional full-text-search index."""

    database_tables = connection.execute(
        "SELECT name, sql FROM sqlite_master "
        "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    if not database_tables:
        raise UnsupportedKnowledgeSchema("Knowledge database has no application tables")

    fts_tables = {
        str(row["name"]): str(row["sql"] or "")
        for row in database_tables
        if "using fts5" in str(row["sql"] or "").lower()
    }
    schema_candidates: list[tuple[int, KnowledgeSchema]] = []
    for table_record in database_tables:
        table_name = str(table_record["name"])
        if (
            table_name in fts_tables
            or table_name.startswith("fts_")
            or table_name.endswith(("_data", "_idx"))
        ):
            continue

        columns = _table_columns(connection, table_name)
        id_column = _first_matching_column(columns, _ID_COLUMNS)
        title_column = _first_matching_column(columns, _TITLE_COLUMNS)
        content_column = _first_matching_column(columns, _CONTENT_COLUMNS)
        if not (id_column and title_column and content_column):
            continue

        fts_table = _matching_fts_table(table_name, fts_tables)
        fts_columns = _table_columns(connection, fts_table) if fts_table else set()
        # Prefer the assignment's conventional table and then an indexed equivalent.
        suitability_score = (100 if table_name == "artifacts" else 0) + (25 if fts_table else 0)
        schema_candidates.append(
            (
                suitability_score,
                KnowledgeSchema(
                    artifact_table=table_name,
                    id_column=id_column,
                    title_column=title_column,
                    content_column=content_column,
                    summary_column=_first_matching_column(columns, _SUMMARY_COLUMNS),
                    type_column=_first_matching_column(columns, _TYPE_COLUMNS),
                    metadata_column=_first_matching_column(columns, _METADATA_COLUMNS),
                    customer_column=_first_matching_column(columns, _CUSTOMER_COLUMNS),
                    fts_table=fts_table,
                    fts_id_column=id_column if id_column in fts_columns else None,
                ),
            )
        )

    if not schema_candidates:
        available_tables = ", ".join(str(row["name"]) for row in database_tables)
        raise UnsupportedKnowledgeSchema(
            "Could not identify an artifact table with ID, title, and content columns. "
            f"Found: {available_tables}"
        )
    return max(schema_candidates, key=lambda candidate: candidate[0])[1]
