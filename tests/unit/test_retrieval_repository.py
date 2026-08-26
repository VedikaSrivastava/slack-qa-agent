import sqlite3
from pathlib import Path

from knowledge_assistant.retrieval.models import (
    AccountLookupInput,
    ReadArtifactsInput,
    SearchKnowledgeInput,
)
from knowledge_assistant.retrieval.repository import SQLiteKnowledgeRepository


def _database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            artifact_type TEXT,
            metadata_json TEXT
        );
        CREATE VIRTUAL TABLE artifacts_fts USING fts5(
            title, content, content='artifacts', content_rowid='rowid'
        );
        INSERT INTO artifacts VALUES (
            'a1', 'Verdant Bay runbook', 'Approved live patch window is Saturday 02:00 UTC.',
            'runbook', '{"customer":"Verdant Bay"}'
        );
        INSERT INTO artifacts_fts(artifacts_fts) VALUES ('rebuild');
        """
    )
    connection.commit()
    connection.close()


def test_fts_search_and_batch_read_preserve_provenance(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.sqlite"
    _database(path)
    repository = SQLiteKnowledgeRepository(path)

    hits = repository.search(SearchKnowledgeInput(query="Verdant Bay patch", limit=3))
    evidence = repository.read(ReadArtifactsInput(artifact_ids=["a1"]))

    assert hits[0].artifact_id == "a1"
    assert evidence[0].title == "Verdant Bay runbook"
    assert "Saturday" in evidence[0].content


def test_search_input_is_parameterized(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.sqlite"
    _database(path)
    repository = SQLiteKnowledgeRepository(path)

    repository.search(SearchKnowledgeInput(query='" OR 1=1 --', limit=3))

    assert repository.inspect_schema().artifact_table == "artifacts"


def test_structured_account_lookup_uses_allowlisted_relational_filters(tmp_path: Path) -> None:
    path = tmp_path / "accounts.sqlite"
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY, scenario_id TEXT, name TEXT, region TEXT, country TEXT
        );
        CREATE TABLE products (product_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE implementations (customer_id TEXT, product_id TEXT);
        CREATE TABLE scenarios (
            scenario_id TEXT PRIMARY KEY, pain_point TEXT, trigger_event TEXT
        );
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY, scenario_id TEXT, artifact_type TEXT, title TEXT,
            created_at TEXT, summary TEXT, content_text TEXT, metadata_json TEXT
        );
        INSERT INTO products VALUES ('p1', 'Event Nexus');
        INSERT INTO customers VALUES
            ('c1', 's1', 'Search Corp', 'North America West', 'United States'),
            ('c2', 's2', 'Dedupe Corp', 'North America West', 'United States');
        INSERT INTO implementations VALUES ('c1', 'p1'), ('c2', 'p1');
        INSERT INTO scenarios VALUES
            ('s1', 'search relevance degradation after taxonomy changes', 'taxonomy rollout'),
            ('s2', 'workflow deduplication drift during handoffs', 'systems consolidation');
        INSERT INTO artifacts VALUES
            ('a1', 's1', 'support_ticket', 'Search issue', '2026-01-01', 'Search summary',
             'Search evidence', '{}'),
            ('a2', 's2', 'support_ticket', 'Dedupe issue', '2026-01-01', 'Dedupe summary',
             'Dedupe evidence', '{}');
        """
    )
    connection.commit()
    connection.close()
    repository = SQLiteKnowledgeRepository(path)

    evidence = repository.lookup_accounts(
        AccountLookupInput(region="North America West", product="Event Nexus")
    )

    assert [item.artifact_id for item in evidence] == ["a2", "a1"]
    assert '"customer":"Search Corp"' in evidence[1].content
    assert '"pain_point":"workflow deduplication drift during handoffs"' in evidence[0].content
