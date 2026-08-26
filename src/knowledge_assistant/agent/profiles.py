"""Code-reviewed agent profiles used by production and reproducible experiments."""

from __future__ import annotations

from dataclasses import dataclass

EVALUATOR_MODEL_NAME = "gpt-5.6-terra"


@dataclass(frozen=True)
class AgentProfile:
    name: str
    model_name: str
    temperature: float | None
    max_history_turns: int
    max_initial_queries: int
    max_refined_queries: int
    search_limit: int
    max_artifacts: int
    max_retrieval_rounds: int
    max_tool_calls: int


BALANCED_GPT_4_1_MINI = AgentProfile(
    name="balanced-gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    temperature=0,
    max_history_turns=6,
    max_initial_queries=3,
    max_refined_queries=2,
    search_limit=5,
    max_artifacts=8,
    max_retrieval_rounds=2,
    max_tool_calls=8,
)
BALANCED_GPT_5_MINI = AgentProfile(
    name="balanced-gpt-5-mini",
    model_name="gpt-5-mini",
    temperature=None,
    max_history_turns=6,
    max_initial_queries=3,
    max_refined_queries=2,
    search_limit=5,
    max_artifacts=8,
    max_retrieval_rounds=2,
    max_tool_calls=8,
)
BALANCED_GPT_5_6_LUNA = AgentProfile(
    name="balanced-gpt-5.6-luna",
    model_name="gpt-5.6-luna",
    temperature=None,
    max_history_turns=6,
    max_initial_queries=3,
    max_refined_queries=2,
    search_limit=5,
    max_artifacts=8,
    max_retrieval_rounds=2,
    max_tool_calls=8,
)
LEAN_GPT_4_1_MINI = AgentProfile(
    name="lean-gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    temperature=0,
    max_history_turns=6,
    max_initial_queries=2,
    max_refined_queries=1,
    search_limit=4,
    max_artifacts=6,
    max_retrieval_rounds=1,
    max_tool_calls=4,
)
WIDE_GPT_4_1_MINI = AgentProfile(
    name="wide-gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    temperature=0,
    max_history_turns=6,
    max_initial_queries=3,
    max_refined_queries=2,
    search_limit=8,
    max_artifacts=8,
    max_retrieval_rounds=2,
    max_tool_calls=8,
)

PRODUCTION_PROFILE = BALANCED_GPT_4_1_MINI
EXPERIMENT_PROFILES = {
    profile.name: profile
    for profile in (
        BALANCED_GPT_4_1_MINI,
        BALANCED_GPT_5_MINI,
        BALANCED_GPT_5_6_LUNA,
        LEAN_GPT_4_1_MINI,
        WIDE_GPT_4_1_MINI,
    )
}


def get_experiment_profile(name: str) -> AgentProfile:
    try:
        return EXPERIMENT_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(EXPERIMENT_PROFILES)
        raise ValueError(f"Unknown experiment profile {name!r}; choose one of: {choices}") from exc
