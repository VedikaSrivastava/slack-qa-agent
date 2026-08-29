import pytest
from pydantic import ValidationError

from knowledge_assistant.retrieval.models import (
    AccountLookupInput,
    ReadArtifactsInput,
    SearchKnowledgeInput,
)


def test_search_limit_has_hard_upper_bound() -> None:
    with pytest.raises(ValidationError):
        SearchKnowledgeInput(query="customer", limit=11)


def test_artifact_batch_has_hard_upper_bound() -> None:
    with pytest.raises(ValidationError):
        ReadArtifactsInput(artifact_ids=[str(index) for index in range(9)])


def test_account_lookup_requires_a_filter_and_has_a_hard_limit() -> None:
    with pytest.raises(ValidationError):
        AccountLookupInput()
    with pytest.raises(ValidationError):
        AccountLookupInput(country="Canada", limit=17)
