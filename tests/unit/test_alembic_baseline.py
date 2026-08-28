from __future__ import annotations

import runpy
from pathlib import Path
from typing import Any

import pytest
from alembic import command
from alembic.config import Config

from knowledge_assistant.config import get_database_settings

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
VERSIONS_DIRECTORY = REPOSITORY_ROOT / "alembic" / "versions"


def test_unreleased_schema_has_one_baseline_revision() -> None:
    revision_files = sorted(
        path for path in VERSIONS_DIRECTORY.glob("*.py") if path.name != "__init__.py"
    )

    assert [path.name for path in revision_files] == ["0001_initial_runtime_schema.py"]
    migration: dict[str, Any] = runpy.run_path(str(revision_files[0]))
    assert migration["revision"] == "0001"
    assert migration["down_revision"] is None


def test_baseline_compiles_complete_postgres_upgrade_sql(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv(
        "DATABASE_URL",
        "postgresql+asyncpg://migration_user:migration_password@localhost/runtime",
    )
    get_database_settings.cache_clear()
    try:
        config = Config(str(REPOSITORY_ROOT / "alembic.ini"))
        command.upgrade(config, "head", sql=True)
    finally:
        get_database_settings.cache_clear()

    sql = capsys.readouterr().out
    assert "CREATE TABLE agent_runs" in sql
    assert "CREATE TABLE slack_turns" in sql
    assert "CREATE TABLE run_sources" in sql
    assert "CREATE TABLE slack_stop_events" in sql
    assert "CREATE TABLE slack_stopped_streams" in sql
    assert "CREATE TABLE run_delivery_parts" in sql
    assert "CREATE TABLE feedback" not in sql
    assert "slack_placeholder_ts" not in sql
    assert "inngest_event_id" not in sql
    assert "CREATE INDEX ix_slack_turns_causal_head" in sql
    assert "CREATE UNIQUE INDEX uq_slack_turns_processing_conversation" in sql
