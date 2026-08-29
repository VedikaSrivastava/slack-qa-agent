"""End-to-end evaluation of agent-owned Slack thread follow-ups through LangGraph."""

from __future__ import annotations

import json
import time
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import anyio
from pydantic import BaseModel, Field, TypeAdapter

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.agent.profiles import AgentProfile
from knowledge_assistant.agent.responder import create_responder_classifier
from knowledge_assistant.application.question_processor import QuestionProcessor
from knowledge_assistant.config import (
    APPLICATION_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_VERSION,
    AgentRuntimeSettings,
)
from knowledge_assistant.evals._stats import percentile as _percentile
from knowledge_assistant.evals._stats import sum_optional as _sum_optional
from knowledge_assistant.evals.datasets import dataset_digest
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.models import CheckResult, EvalCase
from knowledge_assistant.integrations.slack.routing import (
    ResponderClassificationRequest,
    ResponderClassifier,
    ResponderDecision,
    ResponderPromptVariant,
    RoutingAction,
    SlackThreadIdentity,
    decide_responder_classification,
)

GRAPH_FOLLOW_UP_ROUTING_SUITE = "graph_follow_up_routing"
GRAPH_FOLLOW_UP_ROUTING_CASES_PATH = (
    Path(__file__).with_name("cases") / "graph_follow_up_routing.json"
)


class GraphFollowUpRoutingEvalCase(EvalCase):
    """One new Slack thread followed by an unmentioned reply in that same thread."""

    initial_question: str = Field(min_length=1, max_length=8_000)
    follow_up: str = Field(min_length=1, max_length=8_000)
    expected_action: RoutingAction


class FollowUpAnswerEvaluation(BaseModel):
    """Safe answer-quality summary; never persist the raw generated answer."""

    passed: bool
    checks: list[CheckResult]
    source_ids: list[str]
    retrieved_artifact_ids: list[str]
    tool_call_count: int
    model_call_count: int
    retrieval_round_count: int
    input_tokens: int | None
    output_tokens: int | None


class GraphActionMetrics(BaseModel):
    """Per-graph-invocation cost data without persisting generated answer text."""

    tool_call_count: int
    model_call_count: int
    retrieval_round_count: int
    input_tokens: int | None
    output_tokens: int | None


class GraphFollowUpRoutingEvalResult(BaseModel):
    """Routing and graph outcomes for a single agent-owned Slack thread."""

    case_id: str
    category: str
    expected_action: RoutingAction
    classification: ResponderDecision
    actual_action: RoutingAction
    routing_passed: bool
    initial_answer_duration_ms: int = Field(ge=0)
    initial_answer: GraphActionMetrics
    routing_duration_ms: int = Field(ge=0)
    follow_up_answer_duration_ms: int | None = Field(default=None, ge=0)
    graph_invoked_for_follow_up: bool
    follow_up_answer: FollowUpAnswerEvaluation | None = None


def load_graph_follow_up_routing_cases() -> list[GraphFollowUpRoutingEvalCase]:
    """Load the dedicated non-gold end-to-end follow-up dataset."""

    return TypeAdapter(list[GraphFollowUpRoutingEvalCase]).validate_json(
        GRAPH_FOLLOW_UP_ROUTING_CASES_PATH.read_bytes()
    )


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _answer_evaluation(
    case: GraphFollowUpRoutingEvalCase, response: AgentResponse
) -> FollowUpAnswerEvaluation:
    evaluated = evaluate_response(
        EvalCase(
            id=case.id,
            category=case.category,
            question=case.follow_up,
            reference_answer=case.reference_answer,
            expected_facts=case.expected_facts,
            expected_entities=case.expected_entities,
            expected_dates=case.expected_dates,
            expected_commands=case.expected_commands,
            expected_customers=case.expected_customers,
            expected_source_ids=case.expected_source_ids,
            expected_show_sources=case.expected_show_sources,
            forbidden_phrases=case.forbidden_phrases,
            max_tool_calls=case.max_tool_calls,
            max_model_calls=case.max_model_calls,
            max_retrieval_rounds=case.max_retrieval_rounds,
            insufficient_evidence_acceptable=case.insufficient_evidence_acceptable,
        ),
        response,
    )
    return FollowUpAnswerEvaluation(
        passed=evaluated.passed,
        checks=evaluated.checks,
        source_ids=evaluated.source_ids,
        retrieved_artifact_ids=evaluated.retrieved_artifact_ids,
        tool_call_count=evaluated.tool_call_count,
        model_call_count=evaluated.model_call_count,
        retrieval_round_count=evaluated.retrieval_round_count,
        input_tokens=evaluated.input_tokens,
        output_tokens=evaluated.output_tokens,
    )


def _action_metrics(response: AgentResponse) -> GraphActionMetrics:
    return GraphActionMetrics(
        tool_call_count=response.tool_call_count,
        model_call_count=response.model_call_count,
        retrieval_round_count=response.retrieval_round_count,
        input_tokens=response.input_tokens,
        output_tokens=response.output_tokens,
    )


async def run_graph_follow_up_routing_cases(
    processor: QuestionProcessor,
    classifier: ResponderClassifier,
    cases: list[GraphFollowUpRoutingEvalCase],
    *,
    prompt_variant: ResponderPromptVariant,
) -> list[GraphFollowUpRoutingEvalResult]:
    """Run real graph setup and follow-up calls; silent replies must not invoke the graph."""

    results: list[GraphFollowUpRoutingEvalResult] = []
    for case in cases:
        conversation_id = f"eval:follow-up:{case.id}:{uuid.uuid4().hex[:8]}"
        started_at = time.perf_counter()
        initial_response = await processor.answer(
            question=case.initial_question,
            conversation_id=conversation_id,
            agent_run_id=str(uuid.uuid4()),
        )
        initial_answer_duration_ms = max(0, int((time.perf_counter() - started_at) * 1_000))
        clarification_question = (
            initial_response.answer[:8_000].strip()
            if initial_response.requires_user_input
            else None
        )
        classification_request = ResponderClassificationRequest(
            thread=SlackThreadIdentity(
                team_id="eval-team",
                channel_id="eval-channel",
                # Slack timestamps are capped at 64 characters; the longer conversation ID is
                # used only by LangGraph and remains separate from this routing identity.
                thread_ts=f"eval-{uuid.uuid4().hex}",
            ),
            user_id="eval-user",
            message_text=case.follow_up,
            last_agent_clarification_question=clarification_question,
            last_agent_response=(
                initial_response.answer[:8_000].strip()
                if prompt_variant is ResponderPromptVariant.LATEST_AGENT_CONTEXT
                else None
            ),
        )
        started_at = time.perf_counter()
        classification = await classifier.classify(classification_request)
        routing_duration_ms = max(0, int((time.perf_counter() - started_at) * 1_000))
        action = decide_responder_classification(classification).action
        follow_up_answer: FollowUpAnswerEvaluation | None = None
        follow_up_answer_duration_ms: int | None = None
        if action is RoutingAction.RESPOND:
            started_at = time.perf_counter()
            follow_up_response = await processor.answer(
                question=case.follow_up,
                conversation_id=conversation_id,
                agent_run_id=str(uuid.uuid4()),
            )
            follow_up_answer_duration_ms = max(0, int((time.perf_counter() - started_at) * 1_000))
            if case.expected_action is RoutingAction.RESPOND:
                follow_up_answer = _answer_evaluation(case, follow_up_response)
        results.append(
            GraphFollowUpRoutingEvalResult(
                case_id=case.id,
                category=case.category,
                expected_action=case.expected_action,
                classification=classification.decision,
                actual_action=action,
                routing_passed=action is case.expected_action,
                initial_answer_duration_ms=initial_answer_duration_ms,
                initial_answer=_action_metrics(initial_response),
                routing_duration_ms=routing_duration_ms,
                follow_up_answer_duration_ms=follow_up_answer_duration_ms,
                graph_invoked_for_follow_up=action is RoutingAction.RESPOND,
                follow_up_answer=follow_up_answer,
            )
        )
    return results


async def run_graph_follow_up_routing_suite(
    settings: AgentRuntimeSettings,
    profile: AgentProfile,
    cases: list[GraphFollowUpRoutingEvalCase],
    prompt_variant: ResponderPromptVariant,
) -> list[GraphFollowUpRoutingEvalResult]:
    """Create the production graph and classifier once, then evaluate fresh Slack threads."""

    from knowledge_assistant.agent.processor import create_question_processor
    from knowledge_assistant.evals.harness import EVAL_MAX_RETRIES, new_eval_checkpointer

    classifier = create_responder_classifier(
        settings, profile, prompt_variant=prompt_variant, max_retries=EVAL_MAX_RETRIES
    )
    async with create_question_processor(
        settings,
        profile,
        checkpointer=new_eval_checkpointer(),
        max_retries=EVAL_MAX_RETRIES,
    ) as processor:
        return await run_graph_follow_up_routing_cases(
            processor,
            classifier,
            cases,
            prompt_variant=prompt_variant,
        )


def graph_follow_up_routing_metrics(
    results: list[GraphFollowUpRoutingEvalResult],
) -> dict[str, object]:
    """Measure safe routing plus quality, latency, and action cost of accepted follow-ups."""

    total = len(results)
    expected_respond = [
        result for result in results if result.expected_action is RoutingAction.RESPOND
    ]
    actual_respond = [result for result in results if result.actual_action is RoutingAction.RESPOND]
    true_positives = sum(
        result.expected_action is RoutingAction.RESPOND
        and result.actual_action is RoutingAction.RESPOND
        for result in results
    )
    false_positives = sum(
        result.expected_action is RoutingAction.STAY_SILENT
        and result.actual_action is RoutingAction.RESPOND
        for result in results
    )
    false_negatives = sum(
        result.expected_action is RoutingAction.RESPOND
        and result.actual_action is RoutingAction.STAY_SILENT
        for result in results
    )
    precision = _ratio(true_positives, len(actual_respond))
    recall = _ratio(true_positives, len(expected_respond))
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and precision + recall
        else None
    )
    category_totals = Counter(result.category for result in results)
    category_passes = Counter(result.category for result in results if result.routing_passed)
    accepted_answers = [
        result.follow_up_answer for result in results if result.follow_up_answer is not None
    ]
    initial_answers = [result.initial_answer for result in results]
    setup_durations = [result.initial_answer_duration_ms for result in results]
    routing_durations = [result.routing_duration_ms for result in results]
    answer_durations = [
        result.follow_up_answer_duration_ms
        for result in results
        if result.follow_up_answer_duration_ms is not None
    ]
    total_durations = [
        result.initial_answer_duration_ms
        + result.routing_duration_ms
        + (result.follow_up_answer_duration_ms or 0)
        for result in results
    ]
    return {
        "routing": {
            "action_accuracy": _ratio(sum(result.routing_passed for result in results), total),
            "respond_precision": precision,
            "respond_recall": recall,
            "respond_f1": f1,
            "unwanted_interruptions": {
                "count": false_positives,
                "rate": _ratio(false_positives, total - len(expected_respond)),
            },
            "missed_follow_ups": {
                "count": false_negatives,
                "rate": _ratio(false_negatives, len(expected_respond)),
            },
            "category_accuracy": {
                category: category_passes[category] / count
                for category, count in sorted(category_totals.items())
            },
        },
        "graph": {
            "follow_up_invocations": len(actual_respond),
            "unexpected_follow_up_invocations": false_positives,
            "accepted_follow_up_answer_pass_rate": _ratio(
                sum(answer.passed for answer in accepted_answers),
                len(accepted_answers),
            ),
            "end_to_end_positive_pass_rate": _ratio(
                sum(
                    result.routing_passed
                    and result.follow_up_answer is not None
                    and result.follow_up_answer.passed
                    for result in expected_respond
                ),
                len(expected_respond),
            ),
            "follow_up_tool_calls": sum(answer.tool_call_count for answer in accepted_answers),
            "follow_up_model_calls": sum(answer.model_call_count for answer in accepted_answers),
            "follow_up_input_tokens": _sum_optional(
                [answer.input_tokens for answer in accepted_answers]
            ),
            "follow_up_output_tokens": _sum_optional(
                [answer.output_tokens for answer in accepted_answers]
            ),
            "initial_tool_calls": sum(answer.tool_call_count for answer in initial_answers),
            "initial_model_calls": sum(answer.model_call_count for answer in initial_answers),
            "initial_input_tokens": _sum_optional(
                [answer.input_tokens for answer in initial_answers]
            ),
            "initial_output_tokens": _sum_optional(
                [answer.output_tokens for answer in initial_answers]
            ),
            "routing_classifier_model_calls": total,
            "routing_classifier_tokens": "not exposed by the production structured classifier",
            "thread_total_tool_calls": sum(answer.tool_call_count for answer in initial_answers)
            + sum(answer.tool_call_count for answer in accepted_answers),
            "thread_total_model_calls": total
            + sum(answer.model_call_count for answer in initial_answers)
            + sum(answer.model_call_count for answer in accepted_answers),
        },
        "latency_ms": {
            "initial_answer_p50": _percentile(setup_durations, 0.50),
            "initial_answer_p99": _percentile(setup_durations, 0.99),
            "routing_p50": _percentile(routing_durations, 0.50),
            "routing_p99": _percentile(routing_durations, 0.99),
            "accepted_follow_up_answer_p50": _percentile(answer_durations, 0.50),
            "accepted_follow_up_answer_p99": _percentile(answer_durations, 0.99),
            "thread_total_p50": _percentile(total_durations, 0.50),
            "thread_total_p99": _percentile(total_durations, 0.99),
        },
    }


async def write_graph_follow_up_routing_results(
    output_path: Path,
    *,
    profile: AgentProfile,
    prompt_variant: ResponderPromptVariant,
    cases: list[GraphFollowUpRoutingEvalCase],
    results: list[GraphFollowUpRoutingEvalResult],
) -> None:
    """Write a metrics-focused report without raw prompts, responses, or evidence content."""

    payload = {
        "suite": GRAPH_FOLLOW_UP_ROUTING_SUITE,
        "dataset_digest": dataset_digest(cases),
        "application_version": APPLICATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "profile": asdict(profile),
        "prompt_variant": prompt_variant.value,
        "protocol": {
            "name": "single_run_serial_v1",
            "max_concurrency": 1,
            "fresh_conversation_per_case": True,
        },
        "saved_at": datetime.now(UTC).isoformat(),
        "routing_passed": all(result.routing_passed for result in results),
        "end_to_end_passed": all(
            result.expected_action is RoutingAction.STAY_SILENT
            or (
                result.routing_passed
                and result.follow_up_answer is not None
                and result.follow_up_answer.passed
            )
            for result in results
        ),
        "case_count": len(results),
        "metrics": graph_follow_up_routing_metrics(results),
        "results": [result.model_dump(mode="json") for result in results],
    }
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
