import pytest

from knowledge_assistant.agent.models import AgentResponse, EvidenceReference
from knowledge_assistant.agent.profiles import (
    EXPERIMENT_PROFILES,
    PRODUCTION_PROFILE,
    get_experiment_profile,
)
from knowledge_assistant.evals.augmentation import (
    AUGMENTATION_DATASET_NAME,
    CandidateQuestion,
    build_candidate_case,
)
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.langsmith_dataset import (
    DATASET_NAME,
    DATASET_VERSION_TAG,
    dataset_digest,
)
from knowledge_assistant.evals.langsmith_evaluators import (
    action_budget_pass,
    source_recall,
)
from knowledge_assistant.evals.protocols import (
    CONFIRMATION_PROTOCOL,
    SCREENING_PROTOCOL,
    get_experiment_protocol,
)
from knowledge_assistant.evals.runner import load_cases


def test_full_suite_is_the_immutable_seven_case_official_benchmark() -> None:
    cases = load_cases("full")

    assert len(cases) == 7
    assert len({case.id for case in cases}) == 7
    assert all(case.id.startswith("official-") for case in cases)
    assert all(case.reference_answer for case in cases)
    assert all(case.expected_source_ids for case in cases)
    assert dataset_digest(cases) == dataset_digest(load_cases("full"))
    assert DATASET_VERSION_TAG == "official-v1"


def test_reference_answers_satisfy_every_deterministic_gold_label() -> None:
    for case in load_cases("full"):
        response = AgentResponse(
            answer=case.reference_answer,
            sources=[
                EvidenceReference(artifact_id=source_id, title="gold source")
                for source_id in case.expected_source_ids
            ],
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
        tool_call_count=case.max_tool_calls,
        retrieval_round_count=case.max_retrieval_rounds,
        insufficient_evidence=False,
    )
    outputs = {"response": response.model_dump(mode="json")}
    references = {"case": case.model_dump(mode="json")}

    assert source_recall({}, outputs, references)["score"] == 1.0
    assert action_budget_pass({}, outputs, references)["score"] == 1


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
