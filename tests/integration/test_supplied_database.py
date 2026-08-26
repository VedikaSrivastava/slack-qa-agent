from pathlib import Path

import pytest

from slack_qa_agent.retrieval.models import AccountLookupInput, SearchKnowledgeInput
from slack_qa_agent.retrieval.repository import SQLiteKnowledgeRepository

DATABASE_PATH = Path("data/synthetic_startup.sqlite")
pytestmark = pytest.mark.integration


@pytest.mark.skipif(not DATABASE_PATH.is_file(), reason="supplied database is not available")
def test_supplied_database_schema_and_verdant_bay_retrieval() -> None:
    repository = SQLiteKnowledgeRepository(DATABASE_PATH)

    schema = repository.inspect_schema()
    hits = repository.search(
        SearchKnowledgeInput(query="Verdant Bay approved live patch window", limit=5)
    )
    accounts = repository.lookup_accounts(
        AccountLookupInput(region="North America West", product="Event Nexus")
    )
    canada_accounts = repository.lookup_accounts(
        AccountLookupInput(
            country="Canada",
            product="Orchestrator",
            pain_point_terms=["approval workflow failures"],
        )
    )

    assert schema.artifact_table == "artifacts"
    assert schema.content_column == "content_text"
    assert schema.fts_table == "artifacts_fts"
    assert schema.fts_id_column == "artifact_id"
    assert hits[0].artifact_id == "art_fff67d92fe41"
    assert len(accounts) == 12
    assert {item.artifact_id for item in accounts} == {
        "art_90991e25335f",
        "art_8b0063fbb3cb",
        "art_10f7e8b72e09",
        "art_9345d5653840",
        "art_0ac4efa5a0ff",
        "art_8478ccd5b200",
        "art_2f780acc1f96",
        "art_87b096c2c2d3",
        "art_4eccfd9dcf29",
        "art_0927b1cbb7f4",
        "art_3ba29fe1e026",
        "art_f64972a66eeb",
    }
    assert {item.artifact_id for item in canada_accounts} == {
        "art_f4a8c516b934",
        "art_cbfb5f92862c",
        "art_6be1b68b59cb",
        "art_39c1434aa40a",
        "art_e9c20e0a23e0",
        "art_981952a71434",
        "art_b86a0ca2ce1e",
    }
