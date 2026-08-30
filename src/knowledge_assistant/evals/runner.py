"""Strict local evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import anyio
from pydantic import TypeAdapter

from knowledge_assistant.agent.profiles import (
    EXPERIMENT_PROFILES,
    AgentProfile,
    get_experiment_profile,
)
from knowledge_assistant.config import (
    APPLICATION_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_VERSION,
    AgentRuntimeSettings,
)
from knowledge_assistant.evals.datasets import (
    EVALUATION_PROTOCOL_VERSION,
    SUITE_CHOICES,
    TAKE_HOME_GOLD_DATASET_DIGEST,
    annotation_digest,
    dataset_digest,
)
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.graph_follow_up_routing import (
    load_graph_follow_up_routing_cases,
    run_graph_follow_up_routing_suite,
    write_graph_follow_up_routing_results,
)
from knowledge_assistant.evals.harness import (
    EVAL_MAX_RETRIES,
    load_eval_agent_settings,
    new_eval_checkpointer,
)
from knowledge_assistant.evals.metrics import suite_metrics
from knowledge_assistant.evals.models import EvalCase, EvalResult
from knowledge_assistant.integrations.slack.routing import ResponderPromptVariant

CASES_DIR = Path(__file__).with_name("cases")


def load_cases(suite: str) -> list[EvalCase]:
    path = CASES_DIR / f"{suite}.json"
    cases = TypeAdapter(list[EvalCase]).validate_json(path.read_bytes())
    if suite == "full" and dataset_digest(cases) != TAKE_HOME_GOLD_DATASET_DIGEST:
        raise ValueError("The immutable take-home gold questions or answers changed")
    return cases


def require_new_report_path(output_path: Path) -> None:
    """Protect prior evaluation evidence from accidental replacement."""

    if output_path.exists():
        raise FileExistsError(f"Evaluation report already exists: {output_path}")


async def run_suite(
    settings: AgentRuntimeSettings,
    suite: str,
    profile: AgentProfile,
    cases: list[EvalCase],
) -> list[EvalResult]:
    # Follow-up routing evaluation does not require the full graph or its optional integrations.
    # Keep this import at the only command boundary that needs the processor.
    from knowledge_assistant.agent.processor import create_question_processor

    results: list[EvalResult] = []
    async with create_question_processor(
        settings,
        profile,
        checkpointer=new_eval_checkpointer(),
        max_retries=EVAL_MAX_RETRIES,
    ) as processor:
        for case in cases:
            conversation_id = f"eval:{suite}:{profile.name}:{case.id}:{uuid.uuid4().hex[:8]}"
            for prior_turn in case.prior_turns:
                await processor.answer(
                    question=prior_turn,
                    conversation_id=conversation_id,
                    agent_run_id=str(uuid.uuid4()),
                )
            started_at = time.perf_counter()
            response = await processor.answer(
                question=case.question,
                conversation_id=conversation_id,
                agent_run_id=str(uuid.uuid4()),
            )
            duration_ms = max(0, int((time.perf_counter() - started_at) * 1_000))
            results.append(evaluate_response(case, response, duration_ms=duration_ms))
    return results


async def write_local_results(
    output_path: Path,
    *,
    suite: str,
    profile: AgentProfile,
    cases: list[EvalCase],
    results: list[EvalResult],
) -> None:
    payload = {
        "status": "completed",
        "suite": suite,
        "dataset_digest": dataset_digest(cases),
        "annotation_digest": annotation_digest(cases),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "application_version": APPLICATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "profile": asdict(profile),
        "strict_contract_passed": all(result.strict_contract_passed for result in results),
        "semantic_quality": "not_judged",
        "case_count": len(results),
        "metrics": suite_metrics(
            results,
            model_name=profile.model_name,
            answer_model_name=profile.answer_model(),
        ),
        "saved_at": datetime.now(UTC).isoformat(),
        "results": [result.model_dump(mode="json") for result in results],
    }
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


async def write_failed_local_results(
    output_path: Path,
    *,
    suite: str,
    profile: AgentProfile,
    cases: list[EvalCase],
    error: Exception,
) -> None:
    """Preserve a safe record of an attempted run without provider details or secrets."""

    payload = {
        "status": "failed",
        "suite": suite,
        "dataset_digest": dataset_digest(cases),
        "annotation_digest": annotation_digest(cases),
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "application_version": APPLICATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "profile": asdict(profile),
        "case_count": 0,
        "saved_at": datetime.now(UTC).isoformat(),
        "error": {
            "code": "evaluation_run_failed",
            "exception_class": type(error).__name__,
        },
        "results": [],
    }
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m knowledge_assistant.evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run deterministic local evaluation")
    run_parser.add_argument("--suite", choices=SUITE_CHOICES, required=True)
    run_parser.add_argument("--profile", choices=tuple(EXPERIMENT_PROFILES), required=True)
    run_parser.add_argument("--env-file", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)

    graph_follow_up_parser = subparsers.add_parser(
        "follow-up-workflow",
        help="Evaluate agent-owned thread follow-ups through the production LangGraph processor",
    )
    graph_follow_up_parser.add_argument(
        "--profile", choices=tuple(EXPERIMENT_PROFILES), required=True
    )
    graph_follow_up_parser.add_argument(
        "--prompt-variant",
        choices=tuple(variant.value for variant in ResponderPromptVariant),
        required=True,
    )
    graph_follow_up_parser.add_argument("--env-file", type=Path)
    graph_follow_up_parser.add_argument("--output", type=Path, required=True)

    # Imported here to avoid a module-load cycle: these import run_suite/load_cases from here.
    from knowledge_assistant.evals.judge import add_judge_subcommand
    from knowledge_assistant.evals.matrix import add_matrix_subcommand

    add_matrix_subcommand(subparsers)
    add_judge_subcommand(subparsers)

    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "run":
        require_new_report_path(args.output)
        profile = get_experiment_profile(args.profile)
        cases = await anyio.to_thread.run_sync(load_cases, args.suite)
        try:
            agent_settings = load_eval_agent_settings(args.env_file)
            results = await run_suite(agent_settings, args.suite, profile, cases)
        except Exception as error:
            await write_failed_local_results(
                args.output,
                suite=args.suite,
                profile=profile,
                cases=cases,
                error=error,
            )
            raise
        await write_local_results(
            args.output,
            suite=args.suite,
            profile=profile,
            cases=cases,
            results=results,
        )
        print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
        # A completed diagnostic run is successful even when the agent misses quality targets.
        # Non-zero exits are reserved for execution or report-integrity failures.
        return 0

    if args.command == "follow-up-workflow":
        require_new_report_path(args.output)
        agent_settings = load_eval_agent_settings(args.env_file)
        profile = get_experiment_profile(args.profile)
        prompt_variant = ResponderPromptVariant(args.prompt_variant)
        graph_follow_up_cases = await anyio.to_thread.run_sync(load_graph_follow_up_routing_cases)
        graph_follow_up_results = await run_graph_follow_up_routing_suite(
            agent_settings,
            profile,
            graph_follow_up_cases,
            prompt_variant,
        )
        await write_graph_follow_up_routing_results(
            args.output,
            profile=profile,
            prompt_variant=prompt_variant,
            cases=graph_follow_up_cases,
            results=graph_follow_up_results,
        )
        print(
            json.dumps(
                [result.model_dump(mode="json") for result in graph_follow_up_results], indent=2
            )
        )
        return 0 if all(result.routing_passed for result in graph_follow_up_results) else 1

    if args.command == "matrix":
        from knowledge_assistant.evals.matrix import normalize_matrix_args, run_matrix

        return await run_matrix(normalize_matrix_args(args))

    if args.command == "judge":
        from knowledge_assistant.evals.judge import run_judge

        return await run_judge(args)

    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if _requires_windows_selector_loop():
        # Psycopg's async connection implementation requires a selector loop on Windows.
        # Linux containers retain asyncio's standard runtime behavior.
        with asyncio.Runner(loop_factory=asyncio.SelectorEventLoop) as runner:
            return runner.run(async_main(args))
    return asyncio.run(async_main(args))


def _requires_windows_selector_loop() -> bool:
    """Keep platform-specific Psycopg compatibility out of the portable evaluation flow."""

    return sys.platform == "win32"


if __name__ == "__main__":
    raise SystemExit(main())
