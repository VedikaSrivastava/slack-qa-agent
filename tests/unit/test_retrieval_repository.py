import sqlite3
from pathlib import Path

import pytest

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
            scenario_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            content_text TEXT NOT NULL,
            artifact_type TEXT,
            metadata_json TEXT
        );
        CREATE VIRTUAL TABLE artifacts_fts USING fts5(
            artifact_id UNINDEXED, title, summary, content_text
        );
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY, scenario_id TEXT, name TEXT, region TEXT, country TEXT
        );
        CREATE TABLE products (product_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE implementations (customer_id TEXT, product_id TEXT);
        CREATE TABLE scenarios (
            scenario_id TEXT PRIMARY KEY, pain_point TEXT, trigger_event TEXT
        );
        INSERT INTO customers VALUES (
            'c1', 's1', 'City of Verdant Bay', 'North America West', 'Canada'
        );
        INSERT INTO artifacts VALUES (
            'a1', 's1', '2026-01-01', 'Verdant Bay runbook', 'Approved patch window',
            'Approved live patch window is Saturday 02:00 UTC.', 'runbook',
            '{"customer":"Verdant Bay"}'
        );
        INSERT INTO artifacts_fts VALUES (
            'a1', 'Verdant Bay runbook', 'Approved patch window',
            'Approved live patch window is Saturday 02:00 UTC.'
        );
        """
    )
    connection.commit()
    connection.close()


def _diversification_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE artifacts (
            artifact_id TEXT PRIMARY KEY,
            scenario_id TEXT NOT NULL,
            created_at TEXT NOT NULL,
            title TEXT NOT NULL,
            summary TEXT NOT NULL,
            content_text TEXT NOT NULL,
            artifact_type TEXT,
            metadata_json TEXT
        );
        CREATE VIRTUAL TABLE artifacts_fts USING fts5(
            artifact_id UNINDEXED, title, summary, content_text
        );
        CREATE TABLE customers (
            customer_id TEXT PRIMARY KEY, scenario_id TEXT, name TEXT, region TEXT, country TEXT
        );
        CREATE TABLE products (product_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE implementations (customer_id TEXT, product_id TEXT);
        CREATE TABLE scenarios (
            scenario_id TEXT PRIMARY KEY, pain_point TEXT, trigger_event TEXT
        );
        INSERT INTO customers VALUES
            ('customer-a', 's1', 'Dominant Customer', 'Region A', 'Country A'),
            ('customer-b', 's2', 'Other Customer', 'Region B', 'Country B'),
            ('customer-c', 's3', 'Third Customer', 'Region C', 'Country C');
        INSERT INTO artifacts VALUES
            ('a1', 's1', '2026-01-01', 'Dominant one', 'shared outage latency',
             'shared outage latency dominant', 'ticket', '{}'),
            ('a2', 's1', '2026-01-02', 'Dominant two', 'shared outage latency',
             'shared outage latency dominant', 'ticket', '{}'),
            ('a3', 's1', '2026-01-03', 'Dominant three', 'shared outage latency',
             'shared outage latency dominant', 'ticket', '{}'),
            ('a4', 's1', '2026-01-04', 'Dominant four', 'shared outage latency',
             'shared outage latency dominant', 'ticket', '{}'),
            ('b1', 's2', '2026-01-01', 'Other one', 'shared', 'shared', 'ticket', '{}'),
            ('b2', 's2', '2026-01-02', 'Other two', 'shared', 'shared', 'ticket', '{}'),
            ('c1', 's3', '2026-01-01', 'Third one', 'shared', 'shared', 'ticket', '{}'),
            ('c2', 's3', '2026-01-02', 'Third two', 'shared', 'shared', 'ticket', '{}');
        INSERT INTO artifacts_fts
            SELECT artifact_id, title, summary, content_text FROM artifacts;
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
    assert hits[0].scenario_id == "s1"
    assert hits[0].customer_name == "City of Verdant Bay"
    assert evidence[0].title == "Verdant Bay runbook"
    assert evidence[0].scenario_id == "s1"
    assert evidence[0].customer_name == "City of Verdant Bay"
    assert "Saturday" in evidence[0].content


def test_search_input_is_parameterized(tmp_path: Path) -> None:
    path = tmp_path / "knowledge.sqlite"
    _database(path)
    repository = SQLiteKnowledgeRepository(path)

    repository.search(SearchKnowledgeInput(query='" OR 1=1 --', limit=3))

    assert repository.validate_runtime_schema().artifact_table == "artifacts"


def test_fts_search_diversifies_globally_ranked_candidates_by_scenario(tmp_path: Path) -> None:
    path = tmp_path / "diverse.sqlite"
    _diversification_database(path)
    repository = SQLiteKnowledgeRepository(path)

    hits = repository.search(SearchKnowledgeInput(query="shared outage latency", limit=5))

    prefixes = [hit.artifact_id[0] for hit in hits]
    assert len(hits) == 5
    assert set(prefixes) == {"a", "b", "c"}
    assert prefixes.count("a") <= 2


def test_comparison_candidate_search_returns_matched_excerpts_across_scenarios(
    tmp_path: Path,
) -> None:
    path = tmp_path / "comparison-candidates.sqlite"
    _diversification_database(path)
    repository = SQLiteKnowledgeRepository(
        path,
        fts_first_pass_results_per_scenario=None,
    )

    hits = repository.search(
        SearchKnowledgeInput(
            query="shared outage latency",
            purpose="comparison_candidates",
            limit=3,
        )
    )

    assert len(hits) == 3
    assert {hit.scenario_id for hit in hits} == {"s1", "s2", "s3"}
    assert {hit.customer_name for hit in hits} == {
        "Dominant Customer",
        "Other Customer",
        "Third Customer",
    }
    assert all("shared" in hit.snippet.casefold() for hit in hits)


def test_comparison_candidate_snippet_uses_the_fts_match_not_static_summary(
    tmp_path: Path,
) -> None:
    path = tmp_path / "candidate-excerpt.sqlite"
    _database(path)
    repository = SQLiteKnowledgeRepository(path)

    ordinary_hit = repository.search(SearchKnowledgeInput(query="Saturday", limit=1))[0]
    candidate_hit = repository.search(
        SearchKnowledgeInput(
            query="Saturday",
            purpose="comparison_candidates",
            limit=1,
        )
    )[0]

    assert ordinary_hit.snippet == "Approved patch window"
    assert "Saturday" in candidate_hit.snippet


def test_fts_search_can_preserve_global_bm25_for_a_control_profile(tmp_path: Path) -> None:
    path = tmp_path / "global.sqlite"
    _diversification_database(path)
    repository = SQLiteKnowledgeRepository(path, fts_first_pass_results_per_scenario=None)

    hits = repository.search(SearchKnowledgeInput(query="shared outage latency", limit=4))

    assert [hit.artifact_id for hit in hits] == ["a1", "a2", "a3", "a4"]


@pytest.mark.parametrize(("first_pass", "maximum_dominant"), [(1, 3), (2, 2), (3, 3)])
def test_fts_scenario_first_pass_is_a_profile_setting(
    tmp_path: Path,
    first_pass: int,
    maximum_dominant: int,
) -> None:
    path = tmp_path / f"first-pass-{first_pass}.sqlite"
    _diversification_database(path)
    repository = SQLiteKnowledgeRepository(
        path,
        fts_first_pass_results_per_scenario=first_pass,
    )

    hits = repository.search(SearchKnowledgeInput(query="shared outage latency", limit=5))

    assert sum(hit.artifact_id.startswith("a") for hit in hits) <= maximum_dominant


def test_fts_search_backfills_when_only_one_scenario_matches(tmp_path: Path) -> None:
    path = tmp_path / "narrow.sqlite"
    _diversification_database(path)
    repository = SQLiteKnowledgeRepository(path)

    hits = repository.search(SearchKnowledgeInput(query="dominant", limit=4))

    assert len(hits) == 4
    assert all(hit.artifact_id.startswith("a") for hit in hits)


def test_missing_fts_schema_fails_instead_of_falling_back(tmp_path: Path) -> None:
    path = tmp_path / "invalid.sqlite"
    connection = sqlite3.connect(path)
    connection.execute("CREATE TABLE artifacts (artifact_id TEXT, title TEXT, content_text TEXT)")
    connection.commit()
    connection.close()
    repository = SQLiteKnowledgeRepository(path)

    with pytest.raises(RuntimeError, match="schema is missing"):
        repository.search(SearchKnowledgeInput(query="customer"))


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
        CREATE VIRTUAL TABLE artifacts_fts USING fts5(
            artifact_id UNINDEXED, title, summary, content_text
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
            ('a3', 's1', 'runbook', 'Older search runbook', '2025-12-01', 'Runbook summary',
             'Less representative search evidence', '{}'),
            ('a2', 's2', 'support_ticket', 'Dedupe issue', '2026-01-01', 'Dedupe summary',
             'Dedupe evidence', '{}');
        INSERT INTO artifacts_fts VALUES
            ('a1', 'Search issue', 'Search summary', 'Search evidence'),
            ('a3', 'Older search runbook', 'Runbook summary',
             'Less representative search evidence'),
            ('a2', 'Dedupe issue', 'Dedupe summary', 'Dedupe evidence');
        """
    )
    connection.commit()
    connection.close()
    repository = SQLiteKnowledgeRepository(path)

    lookup_result = repository.lookup_accounts(
        AccountLookupInput(
            purpose="enumerate_cohort",
            region="North America West",
            product="Event Nexus",
        )
    )

    assert [item.artifact_id for item in lookup_result.evidence] == ["a2", "a1"]
    assert lookup_result.matched_account_count == 2
    assert lookup_result.is_truncated is False
    assert '"customer":"Search Corp"' in lookup_result.evidence[1].content
    assert (
        '"pain_point":"workflow deduplication drift during handoffs"'
        in lookup_result.evidence[0].content
    )

    limited_result = repository.lookup_accounts(
        AccountLookupInput(
            purpose="enumerate_cohort",
            region="North America West",
            product="Event Nexus",
            limit=1,
        )
    )
    assert [item.artifact_id for item in limited_result.evidence] == ["a2"]
    assert limited_result.matched_account_count == 2
    assert limited_result.is_truncated is True

    broad_region_result = repository.lookup_accounts(
        AccountLookupInput(
            purpose="filter_matches",
            region="North America",
            pain_point_terms=["search pattern"],
        )
    )
    assert [item.artifact_id for item in broad_region_result.evidence] == ["a1"]

    country_over_region_result = repository.lookup_accounts(
        AccountLookupInput(
            purpose="filter_matches",
            region="Model-inferred parent region",
            country="UNITED STATES",
            pain_point_terms=["search-relevance"],
        )
    )
    assert [item.artifact_id for item in country_over_region_result.evidence] == ["a1"]
