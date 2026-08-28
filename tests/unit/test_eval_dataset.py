import hashlib
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

import pytest
from langsmith import Client
from langsmith.utils import LangSmithNotFoundError
from pydantic import ValidationError

from knowledge_assistant.agent.models import AgentResponse, EvidenceReference
from knowledge_assistant.agent.profiles import (
    EXPERIMENT_PROFILES,
    PRODUCTION_PROFILE,
    get_experiment_profile,
)
from knowledge_assistant.evals.augmentation import (
    AUGMENTATION_DATASET_NAME,
    CandidateBatch,
    CandidateQuestion,
    build_candidate_case,
)
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.langsmith_dataset import (
    DATASET_NAME,
    DATASET_VERSION_TAG,
    OfficialDatasetIntegrityError,
    _build_example,
    sync_official_dataset,
)
from knowledge_assistant.evals.langsmith_evaluators import (
    action_budget_pass,
    citation_recall,
    retrieval_recall,
)
from knowledge_assistant.evals.models import EvalCase
from knowledge_assistant.evals.protocols import (
    CONFIRMATION_PROTOCOL,
    SCREENING_PROTOCOL,
    get_experiment_protocol,
)
from knowledge_assistant.evals.runner import CASES_DIR, OFFICIAL_SUITE_SHA256, load_cases


@dataclass
class FakeDataset:
    id: uuid.UUID
    name: str


@dataclass
class FakeDatasetVersion:
    as_of: datetime


@dataclass
class FakeExample:
    id: uuid.UUID
    inputs: dict[str, Any] | None
    outputs: dict[str, Any] | None
    metadata: dict[str, Any] | None


class FakeLangSmithClient:
    def __init__(self, cases: list[EvalCase], *, has_official_tag: bool) -> None:
        self.dataset = FakeDataset(id=uuid.uuid4(), name=DATASET_NAME)
        self.has_official_tag = has_official_tag
        self.examples = [self._to_example(_build_example(case)) for case in cases]
        self.create_examples_calls = 0
        self.tag_updates: list[tuple[datetime, str]] = []
        self.latest_version = FakeDatasetVersion(as_of=datetime(2026, 1, 1, tzinfo=UTC))

    @staticmethod
    def _to_example(payload: dict[str, Any]) -> FakeExample:
        return FakeExample(
            id=payload["id"],
            inputs=payload["inputs"],
            outputs=payload["outputs"],
            metadata=payload["metadata"],
        )

    def has_dataset(self, *, dataset_name: str) -> bool:
        assert dataset_name == DATASET_NAME
        return True

    def read_dataset(self, *, dataset_name: str) -> FakeDataset:
        assert dataset_name == DATASET_NAME
        return self.dataset

    def read_dataset_version(
        self,
        *,
        dataset_id: uuid.UUID,
        tag: str,
    ) -> FakeDatasetVersion:
        assert dataset_id == self.dataset.id
        if tag == DATASET_VERSION_TAG and not self.has_official_tag:
            raise LangSmithNotFoundError("tag is not assigned")
        return self.latest_version

    def list_examples(
        self,
        *,
        dataset_id: uuid.UUID,
        as_of: str | None = None,
        splits: list[str] | None = None,
    ) -> list[FakeExample]:
        assert dataset_id == self.dataset.id
        del as_of, splits
        return list(self.examples)

    def create_examples(
        self,
        *,
        dataset_id: uuid.UUID,
        examples: list[dict[str, Any]],
        max_concurrency: int,
    ) -> dict[str, Any]:
        assert dataset_id == self.dataset.id
        assert max_concurrency == 1
        self.create_examples_calls += 1
        self.examples = [self._to_example(payload) for payload in examples]
        return {
            "count": len(examples),
            "example_ids": [str(example["id"]) for example in examples],
            "as_of": self.latest_version.as_of.isoformat(),
        }

    def update_dataset_tag(
        self,
        *,
        dataset_id: uuid.UUID,
        as_of: datetime,
        tag: str,
    ) -> None:
        assert dataset_id == self.dataset.id
        self.has_official_tag = True
        self.tag_updates.append((as_of, tag))


def test_full_suite_is_the_immutable_seven_case_official_benchmark() -> None:
    cases = load_cases("full")

    assert len(cases) == 7
    assert len({case.id for case in cases}) == 7
    assert all(case.id.startswith("official-") for case in cases)
    assert all(case.reference_answer for case in cases)
    assert all(case.expected_source_ids for case in cases)
    assert hashlib.sha256((CASES_DIR / "full.json").read_bytes()).hexdigest() == (
        OFFICIAL_SUITE_SHA256
    )
    assert DATASET_VERSION_TAG == "official-v1"


def test_existing_official_dataset_tag_is_verified_without_being_moved() -> None:
    cases = load_cases("full")
    fake_client = FakeLangSmithClient(cases, has_official_tag=True)

    summary = sync_official_dataset(cast(Client, fake_client), cases)

    assert summary["example_count"] == 7
    assert fake_client.create_examples_calls == 0
    assert fake_client.tag_updates == []


def test_changed_content_behind_official_tag_fails_without_mutating_dataset() -> None:
    cases = load_cases("full")
    fake_client = FakeLangSmithClient(cases, has_official_tag=True)
    fake_client.examples[0].outputs = {"case": {"id": "changed"}}

    with pytest.raises(OfficialDatasetIntegrityError, match="unexpected outputs"):
        sync_official_dataset(cast(Client, fake_client), cases)

    assert fake_client.create_examples_calls == 0
    assert fake_client.tag_updates == []


def test_first_sync_tags_the_exact_created_dataset_version() -> None:
    cases = load_cases("full")
    fake_client = FakeLangSmithClient([], has_official_tag=False)

    sync_official_dataset(cast(Client, fake_client), cases)

    assert fake_client.create_examples_calls == 1
    assert fake_client.tag_updates == [(fake_client.latest_version.as_of, DATASET_VERSION_TAG)]


def test_reference_answers_satisfy_every_deterministic_gold_label() -> None:
    for case in load_cases("full"):
        response = AgentResponse(
            answer=case.reference_answer,
            sources=[
                EvidenceReference(artifact_id=source_id, title="gold source")
                for source_id in case.expected_source_ids
            ],
            retrieved_artifact_ids=case.expected_source_ids,
            tool_call_count=case.max_tool_calls,
            retrieval_round_count=case.max_retrieval_rounds,
            insufficient_evidence=False,
        )

        result = evaluate_response(case, response)

        assert result.passed, {
            check.name: check.details for check in result.checks if not check.passed
        }


def test_experiment_profiles_are_code_defined_and_unknown_names_fail() -> None:
    assert get_experiment_profile(PRODUCTION_PROFILE.name) is PRODUCTION_PROFILE
    assert {profile.model_name for profile in EXPERIMENT_PROFILES.values()} == {
        "gpt-4.1-mini",
        "gpt-5-mini",
        "gpt-5.6-luna",
    }
    assert all(profile.max_tool_calls > 0 for profile in EXPERIMENT_PROFILES.values())
    with pytest.raises(ValueError, match="Unknown experiment profile"):
        get_experiment_profile("from-an-env-var")


def test_production_evidence_budget_can_cover_every_official_source_set() -> None:
    largest_source_set = max(len(case.expected_source_ids) for case in load_cases("full"))

    assert PRODUCTION_PROFILE.max_artifacts >= largest_source_set


def test_offline_experiment_protocols_are_bounded_and_code_defined() -> None:
    assert get_experiment_protocol("screening") is SCREENING_PROTOCOL
    assert SCREENING_PROTOCOL.repetitions == 1
    assert CONFIRMATION_PROTOCOL.repetitions == 3
    assert SCREENING_PROTOCOL.max_concurrency == 1
    assert CONFIRMATION_PROTOCOL.max_concurrency == 1
    with pytest.raises(ValueError, match="Unknown experiment protocol"):
        get_experiment_protocol("run-until-it-looks-good")


def test_langsmith_metrics_score_sources_and_action_budget() -> None:
    case = load_cases("smoke")[0]
    response = AgentResponse(
        answer=case.reference_answer,
        sources=[
            EvidenceReference(artifact_id=source_id, title="source")
            for source_id in case.expected_source_ids
        ],
        retrieved_artifact_ids=case.expected_source_ids,
        tool_call_count=case.max_tool_calls,
        retrieval_round_count=case.max_retrieval_rounds,
        insufficient_evidence=False,
    )
    outputs = {"response": response.model_dump(mode="json")}
    references = {"case": case.model_dump(mode="json")}

    assert citation_recall({}, outputs, references)["score"] == 1.0
    assert retrieval_recall({}, outputs, references)["score"] == 1.0
    assert action_budget_pass({}, outputs, references)["score"] == 1


def test_langsmith_metrics_distinguish_retrieval_from_citations() -> None:
    case = load_cases("smoke")[0]
    response = AgentResponse(
        answer=case.reference_answer,
        sources=[],
        retrieved_artifact_ids=case.expected_source_ids,
    )
    outputs = {"response": response.model_dump(mode="json")}
    references = {"case": case.model_dump(mode="json")}

    assert retrieval_recall({}, outputs, references)["score"] == 1.0
    assert citation_recall({}, outputs, references)["score"] == 0.0


def test_augmentation_candidates_are_stable_and_separate_from_gold() -> None:
    seed = load_cases("full")[0]
    candidate = CandidateQuestion(
        transformation="multi-turn",
        prior_turns=["We were discussing the taxonomy rollout."],
        question="Which account was affected, and what renewal proof did we offer?",
    )

    first = build_candidate_case(seed, candidate)
    second = build_candidate_case(seed, candidate)

    assert first.id == second.id
    assert first.id.startswith(f"candidate-{seed.id}-")
    assert first.expected_source_ids == seed.expected_source_ids
    assert first.reference_answer == seed.reference_answer
    assert first.question != seed.question
    assert AUGMENTATION_DATASET_NAME != DATASET_NAME


def test_augmentation_batch_rejects_duplicate_question_inputs() -> None:
    with pytest.raises(ValidationError, match="augmentation candidates must be distinct"):
        CandidateBatch(
            candidates=[
                CandidateQuestion(transformation="paraphrase", question="What changed?"),
                CandidateQuestion(transformation="multi-turn", question="  what CHANGED?  "),
            ]
        )
