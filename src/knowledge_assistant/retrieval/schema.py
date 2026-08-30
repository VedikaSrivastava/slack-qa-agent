"""Strict validation for the supplied knowledge-database schema."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass


class UnsupportedKnowledgeSchemaError(RuntimeError):
    """Raised when the supplied database cannot support runtime retrieval."""


@dataclass(frozen=True)
class KnowledgeSchema:
    artifact_table: str = "artifacts"
    id_column: str = "artifact_id"
    title_column: str = "title"
    content_column: str = "content_text"
    summary_column: str = "summary"
    type_column: str = "artifact_type"
    fts_table: str = "artifacts_fts"
    fts_id_column: str = "artifact_id"


SUPPLIED_KNOWLEDGE_SCHEMA = KnowledgeSchema()

_REQUIRED_COLUMNS = {
    "artifacts": {
        "artifact_id",
        "scenario_id",
        "artifact_type",
        "title",
        "created_at",
        "summary",
        "content_text",
        "metadata_json",
    },
    "artifacts_fts": {"artifact_id", "title", "summary", "content_text"},
    "customers": {"customer_id", "scenario_id", "name", "region", "country"},
    "implementations": {"customer_id", "product_id"},
    "products": {"product_id", "name"},
    "scenarios": {"scenario_id", "pain_point", "trigger_event"},
}


def _table_columns(connection: sqlite3.Connection, table_name: str) -> set[str]:
    # `table_name` is never user/model input; callers iterate the fixed schema allowlist above.
    return {
        str(column["name"])
        for column in connection.execute(f'PRAGMA table_info("{table_name}")').fetchall()
    }


def validate_knowledge_schema(connection: sqlite3.Connection) -> KnowledgeSchema:
    """Require the documented assignment schema instead of guessing or degrading retrieval."""

    missing_by_table: dict[str, list[str]] = {}
    for table_name, required_columns in _REQUIRED_COLUMNS.items():
        actual_columns = _table_columns(connection, table_name)
        missing_columns = sorted(required_columns - actual_columns)
        if missing_columns:
            missing_by_table[table_name] = missing_columns

    if missing_by_table:
        details = "; ".join(
            f"{table}: {', '.join(columns)}" for table, columns in missing_by_table.items()
        )
        raise UnsupportedKnowledgeSchemaError(f"Knowledge database schema is missing: {details}")

    fts_definition_row = connection.execute(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'artifacts_fts'"
    ).fetchone()
    fts_definition = str(fts_definition_row["sql"] or "") if fts_definition_row else ""
    if "using fts5" not in fts_definition.lower():
        raise UnsupportedKnowledgeSchemaError("artifacts_fts must be an FTS5 virtual table")

    return SUPPLIED_KNOWLEDGE_SCHEMA
