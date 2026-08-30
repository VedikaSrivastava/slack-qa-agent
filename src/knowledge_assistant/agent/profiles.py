"""Code-reviewed agent profiles: production plus one focused model comparison matrix.

A profile fixes everything one evaluation run holds constant: the model(s), and the budgets
that bound the graph (query fan-out, search depth, evidence size, retrieval rounds, hard
tool/model-call ceilings). Experiments are not a grid sweep -- each profile below tests one
stated hypothesis so a report can attribute a change in accuracy, cost, latency, or tool-call
count to that one change.

The agent has two distinct LLM workloads:

* structured classification -- resolve the question, plan retrieval, grade evidence, verify
  grounding, and (for Slack follow-ups) route. Short, schema-constrained, instruction-following
  matters, deep reasoning does not.
* answer synthesis -- `generate_answer` / `repair_answer`. Multi-hop reasoning, faithful
  grounding, and concision matter here; the hard questions are won or lost in this step.

`AgentProfile` therefore allows per-role model overrides (`resolve_model_name`, `plan_model_name`,
`grade_model_name`, `verify_model_name`, `answer_model_name`, `repair_model_name`,
`router_model_name`). Each falls back to `model_name`, except repair which falls back to the
answer model.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

# Judge model for the optional LLM-graded answer-quality pass.
EVALUATOR_MODEL_NAME = "gpt-5"
OPENAI_REQUEST_TIMEOUT_SECONDS = 60.0
OPENAI_MAX_RETRIES = 0

type ReasoningEffort = Literal["none", "low", "medium", "high", "xhigh", "max"]


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
    max_model_calls: int
    fts_candidate_multiplier: int = 6
    fts_first_pass_results_per_scenario: int | None = 2
    reasoning_effort: ReasoningEffort | None = None
    # Optional per-role overrides. None => use the role's documented fallback.
    answer_model_name: str | None = None
    repair_model_name: str | None = None
    resolve_model_name: str | None = None
    plan_model_name: str | None = None
    grade_model_name: str | None = None
    verify_model_name: str | None = None
    router_model_name: str | None = None

    def answer_model(self) -> str:
        return self.answer_model_name or self.model_name

    def repair_model(self) -> str:
        return self.repair_model_name or self.answer_model()

    def resolve_model(self) -> str:
        return self.resolve_model_name or self.model_name

    def plan_model(self) -> str:
        return self.plan_model_name or self.model_name

    def grade_model(self) -> str:
        return self.grade_model_name or self.model_name

    def verify_model(self) -> str:
        return self.verify_model_name or self.model_name

    def router_model(self) -> str:
        return self.router_model_name or self.model_name

    def _temperature_for(self, model_name: str) -> float | None:
        return None if model_name.startswith("gpt-5") else self.temperature

    def router_temperature(self) -> float | None:
        return self._temperature_for(self.router_model())


# Production remains on the established low-latency baseline. Every model comparison uses the
# same retrieval and action budgets so results isolate model-role changes.
BALANCED_GPT_4_1_MINI = AgentProfile(
    name="balanced-gpt-4.1-mini",
    model_name="gpt-4.1-mini",
    temperature=0,
    max_history_turns=6,
    max_initial_queries=3,
    max_refined_queries=2,
    search_limit=5,
    max_artifacts=16,
    max_retrieval_rounds=2,
    max_tool_calls=8,
    # Covers the graph's longest legal path, including one structured-plan repair and one
    # answer repair, without permitting an unbounded model loop.
    max_model_calls=9,
)

SPLIT_ANSWER_GPT_4_1 = replace(
    BALANCED_GPT_4_1_MINI,
    name="split-answer-gpt-4.1",
    answer_model_name="gpt-4.1",
)

BALANCED_GPT_4_1 = replace(
    BALANCED_GPT_4_1_MINI,
    name="balanced-gpt-4.1",
    model_name="gpt-4.1",
)

BALANCED_GPT_5_5 = replace(
    BALANCED_GPT_4_1_MINI,
    name="balanced-gpt-5.5",
    model_name="gpt-5.5",
    temperature=None,
    reasoning_effort="none",
)

BALANCED_GPT_5_6_TERRA = replace(
    BALANCED_GPT_4_1_MINI,
    name="balanced-gpt-5.6-terra",
    model_name="gpt-5.6-terra",
    temperature=None,
    reasoning_effort="none",
)

SPLIT_GPT_5_6_LUNA_SOL = replace(
    BALANCED_GPT_4_1_MINI,
    name="split-gpt-5.6-luna-sol",
    model_name="gpt-5.6-luna",
    answer_model_name="gpt-5.6-sol",
    router_model_name="gpt-5.6-luna",
    temperature=None,
    reasoning_effort="none",
)

SPLIT_GPT_5_4_HYBRID = replace(
    BALANCED_GPT_4_1_MINI,
    name="split-gpt-5.4-hybrid",
    model_name="gpt-5.4-nano",
    temperature=None,
    reasoning_effort="low",
    resolve_model_name="gpt-5.4-nano",
    plan_model_name="gpt-5.4",
    grade_model_name="gpt-5.4-mini",
    verify_model_name="gpt-5.4-mini",
    answer_model_name="gpt-5.4",
    repair_model_name="gpt-5.4",
    router_model_name="gpt-5.4-nano",
)

MODEL_MATRIX_PROFILES: tuple[AgentProfile, ...] = (
    BALANCED_GPT_4_1_MINI,
    SPLIT_ANSWER_GPT_4_1,
    BALANCED_GPT_4_1,
    BALANCED_GPT_5_5,
    BALANCED_GPT_5_6_TERRA,
    SPLIT_GPT_5_6_LUNA_SOL,
    SPLIT_GPT_5_4_HYBRID,
)

# Same model and graph budgets; only global BM25 versus first-pass diversification changes.
# These are screening profiles, not additional production model choices.
RETRIEVAL_TUNING_PROFILES: tuple[AgentProfile, ...] = (
    replace(
        BALANCED_GPT_4_1_MINI,
        name="retrieval-global-bm25",
        fts_candidate_multiplier=1,
        fts_first_pass_results_per_scenario=None,
    ),
    replace(
        BALANCED_GPT_4_1_MINI,
        name="retrieval-diverse-first-pass-1",
        fts_first_pass_results_per_scenario=1,
    ),
    replace(
        BALANCED_GPT_4_1_MINI,
        name="retrieval-diverse-first-pass-2",
        fts_first_pass_results_per_scenario=2,
    ),
    replace(
        BALANCED_GPT_4_1_MINI,
        name="retrieval-diverse-first-pass-3",
        fts_first_pass_results_per_scenario=3,
    ),
)

PRODUCTION_PROFILE = SPLIT_GPT_5_4_HYBRID

EXPERIMENT_PROFILES: dict[str, AgentProfile] = {
    profile.name: profile for profile in (*MODEL_MATRIX_PROFILES, *RETRIEVAL_TUNING_PROFILES)
}


def get_experiment_profile(name: str) -> AgentProfile:
    try:
        return EXPERIMENT_PROFILES[name]
    except KeyError as exc:
        choices = ", ".join(EXPERIMENT_PROFILES)
        raise ValueError(f"Unknown experiment profile {name!r}; choose one of: {choices}") from exc
