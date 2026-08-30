"""Bounded retrieval request and evidence models."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator

MAX_SEARCH_LIMIT = 10
MAX_ARTIFACT_BATCH = 8
MAX_CONTEXT_CHARS = 24_000
MAX_ACCOUNT_RESULTS = 16
MAX_SEARCH_QUERY_CHARS = 1_000
MAX_PAIN_POINT_TERM_CHARS = 128

type JsonValue = bool | int | float | str | list[JsonValue] | dict[str, JsonValue] | None
SearchQueryText = Annotated[
    str,
    Field(min_length=1, max_length=MAX_SEARCH_QUERY_CHARS),
]
PainPointTerm = Annotated[
    str,
    Field(min_length=1, max_length=MAX_PAIN_POINT_TERM_CHARS),
]


class SearchFilters(BaseModel):
    artifact_type: str | None = Field(default=None, min_length=1, max_length=128)


class SearchKnowledgeInput(BaseModel):
    query: SearchQueryText
    purpose: Literal["evidence", "comparison_candidates"] = "evidence"
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=5, ge=1, le=MAX_SEARCH_LIMIT)

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must contain searchable text")
        return normalized


class ReadArtifactsInput(BaseModel):
    artifact_ids: list[str] = Field(min_length=1, max_length=MAX_ARTIFACT_BATCH)
    max_context_chars: int = Field(default=MAX_CONTEXT_CHARS, ge=1_000, le=MAX_CONTEXT_CHARS)

    @field_validator("artifact_ids")
    @classmethod
    def unique_ids(cls, values: list[str]) -> list[str]:
        if any(not value.strip() or len(value) > 512 for value in values):
            raise ValueError("artifact IDs must be non-empty and at most 512 characters")
        return list(dict.fromkeys(values))


class AccountLookupInput(BaseModel):
    """Allowlisted relational filters for cross-account questions."""

    purpose: Literal["filter_matches", "enumerate_cohort"] = Field(
        description=(
            "Use enumerate_cohort when every account in a region/country/product population "
            "must be returned before grouping or comparison; use filter_matches when the "
            "pain point itself is an input constraint."
        ),
    )
    region: str | None = Field(default=None, min_length=1, max_length=128)
    country: str | None = Field(default=None, min_length=1, max_length=128)
    product: str | None = Field(default=None, min_length=1, max_length=128)
    pain_point_terms: list[PainPointTerm] = Field(default_factory=list, max_length=4)
    limit: int = Field(default=MAX_ACCOUNT_RESULTS, ge=1, le=MAX_ACCOUNT_RESULTS)

    @field_validator("pain_point_terms")
    @classmethod
    def normalize_terms(cls, values: list[str]) -> list[str]:
        normalized = [" ".join(value.split()) for value in values]
        if any(not value or len(value) > MAX_PAIN_POINT_TERM_CHARS for value in normalized):
            raise ValueError(
                "pain point terms must be non-empty and at most "
                f"{MAX_PAIN_POINT_TERM_CHARS} characters"
            )
        return list(dict.fromkeys(normalized))

    @model_validator(mode="after")
    def require_filter(self) -> AccountLookupInput:
        if not any((self.region, self.country, self.product, self.pain_point_terms)):
            raise ValueError("at least one structured account filter is required")
        if self.purpose == "enumerate_cohort":
            if self.pain_point_terms:
                raise ValueError("cohort enumeration cannot filter on output pain-point categories")
            if not any((self.region, self.country, self.product)):
                raise ValueError("cohort enumeration requires a region, country, or product")
        return self


class SearchHit(BaseModel):
    artifact_id: str = Field(min_length=1, max_length=512)
    scenario_id: str | None = Field(default=None, min_length=1, max_length=512)
    customer_name: str | None = Field(default=None, min_length=1, max_length=1_000)
    title: str = Field(min_length=1, max_length=1_000)
    artifact_type: str | None = Field(default=None, max_length=128)
    snippet: str = Field(max_length=500)
    score: float | None = None
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class EvidenceItem(SearchHit):
    content: str = Field(max_length=MAX_CONTEXT_CHARS)
    retrieval_origin: Literal["lexical", "search_excerpt", "account_lookup"] = "lexical"


class AccountLookupResult(BaseModel):
    """Bounded account evidence plus deterministic population coverage."""

    evidence: list[EvidenceItem] = Field(max_length=MAX_ACCOUNT_RESULTS)
    matched_account_count: int = Field(ge=0)
    is_truncated: bool

    @model_validator(mode="after")
    def validate_coverage(self) -> AccountLookupResult:
        returned_account_count = len(self.evidence)
        if returned_account_count > self.matched_account_count:
            raise ValueError("returned accounts cannot exceed the matched population")
        if self.is_truncated is not (returned_account_count < self.matched_account_count):
            raise ValueError("account lookup truncation must agree with population counts")
        return self


class AccountLookupCoverage(BaseModel):
    """Serializable lookup scope supplied to evidence grading and answer verification."""

    purpose: Literal["filter_matches", "enumerate_cohort"]
    matched_account_count: int = Field(ge=0)
    returned_account_count: int = Field(ge=0)
    is_truncated: bool

    @model_validator(mode="after")
    def validate_counts(self) -> AccountLookupCoverage:
        if self.returned_account_count > self.matched_account_count:
            raise ValueError("returned accounts cannot exceed the matched population")
        if self.is_truncated is not (self.returned_account_count < self.matched_account_count):
            raise ValueError("account lookup truncation must agree with population counts")
        return self
