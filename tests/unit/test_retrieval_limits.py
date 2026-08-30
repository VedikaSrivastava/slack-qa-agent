import pytest
from pydantic import ValidationError

from knowledge_assistant.retrieval.models import (
    MAX_PAIN_POINT_TERM_CHARS,
    MAX_SEARCH_QUERY_CHARS,
    AccountLookupInput,
    EvidenceItem,
    ReadArtifactsInput,
    SearchKnowledgeInput,
)


def test_search_limit_has_hard_upper_bound() -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeInput(query="customer", limit=11)


def test_search_query_schema_has_the_runtime_character_bound() -> None:
    schema = SearchKnowledgeInput.model_json_schema()

    assert schema["properties"]["query"]["maxLength"] == MAX_SEARCH_QUERY_CHARS


def test_search_rejects_an_unknown_purpose() -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeInput.model_validate({"query": "customer", "purpose": "unbounded"})


def test_search_excerpt_is_a_typed_evidence_origin() -> None:
    evidence = EvidenceItem(
        artifact_id="artifact-a",
        scenario_id="scenario-a",
        customer_name="Customer A",
        title="Matched excerpt",
        snippet="candidate evidence",
        content="candidate evidence",
        retrieval_origin="search_excerpt",
    )

    assert evidence.retrieval_origin == "search_excerpt"


def test_artifact_batch_has_hard_upper_bound() -> None:
    with pytest.raises(ValidationError):
        ReadArtifactsInput(artifact_ids=[str(index) for index in range(9)])


def test_account_lookup_requires_a_filter_and_has_a_hard_limit() -> None:
    with pytest.raises(ValidationError):
        AccountLookupInput(purpose="filter_matches")
    with pytest.raises(ValidationError):
        AccountLookupInput(purpose="filter_matches", country="Canada", limit=17)


def test_account_lookup_schema_exposes_the_pain_point_term_character_limit() -> None:
    schema = AccountLookupInput.model_json_schema()

    assert (
        schema["properties"]["pain_point_terms"]["items"]["maxLength"] == MAX_PAIN_POINT_TERM_CHARS
    )


def test_cohort_enumeration_rejects_output_category_filters() -> None:
    with pytest.raises(ValidationError, match="output pain-point categories"):
        AccountLookupInput(
            purpose="enumerate_cohort",
            region="North America West",
            pain_point_terms=["duplicate action"],
        )
