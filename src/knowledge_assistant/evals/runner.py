"""Strict local and LangSmith evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import anyio
from langsmith import Client
from pydantic import TypeAdapter

from knowledge_assistant.agent.processor import create_question_processor
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
    AugmentationSettings,
    EvaluationSettings,
    LangSmithSettings,
)
from knowledge_assistant.evals.augmentation import generate_augmentation_candidates
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.langsmith_dataset import dataset_digest, sync_official_dataset
from knowledge_assistant.evals.langsmith_experiment import run_official_benchmark
from knowledge_assistant.evals.models import EvalCase, EvalResult
from knowledge_assistant.evals.protocols import EXPERIMENT_PROTOCOLS, get_experiment_protocol

CASES_DIR = Path(__file__).with_name("cases")
OFFICIAL_SUITE_SHA256 = "1b610b009df6588d0803230a5d35f0dff371ed11480e8b9ba42f84a2b79a16b6"


def load_cases(suite: str) -> list[EvalCase]:
    path = CASES_DIR / f"{suite}.json"
    return TypeAdapter(list[EvalCase]).validate_json(path.read_bytes())


def _validate_env_file(env_file: Path | None) -> None:
    if env_file is not None and not env_file.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {env_file}")


def load_agent_settings(env_file: Path | None) -> AgentRuntimeSettings:
    _validate_env_file(env_file)
    return AgentRuntimeSettings(_env_file=env_file)


def load_evaluation_settings(env_file: Path | None) -> EvaluationSettings:
    _validate_env_file(env_file)
    return EvaluationSettings(_env_file=env_file)


def load_langsmith_settings(env_file: Path | None) -> LangSmithSettings:
    _validate_env_file(env_file)
    return LangSmithSettings(_env_file=env_file)


def load_augmentation_settings(env_file: Path | None) -> AugmentationSettings:
    _validate_env_file(env_file)
    return AugmentationSettings(_env_file=env_file)


def create_langsmith_client(
    settings: LangSmithSettings | EvaluationSettings,
) -> Client:
    return Client(api_key=settings.langsmith_api_key.get_secret_value())


async def run_suite(
    settings: AgentRuntimeSettings,
    suite: str,
    profile: AgentProfile,
    cases: list[EvalCase],
) -> list[EvalResult]:
    results: list[EvalResult] = []
    async with create_question_processor(settings, profile) as processor:
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
        "suite": suite,
        "dataset_digest": dataset_digest(cases),
        "application_version": APPLICATION_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "profile": asdict(profile),
        "passed": all(result.passed for result in results),
        "case_count": len(results),
        "saved_at": datetime.now(UTC).isoformat(),
        "results": [result.model_dump(mode="json") for result in results],
    }
    async_output_path = anyio.Path(output_path)
    await async_output_path.parent.mkdir(parents=True, exist_ok=True)
    await async_output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m knowledge_assistant.evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run deterministic local evaluation")
    run_parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    run_parser.add_argument("--profile", choices=tuple(EXPERIMENT_PROFILES), required=True)
    run_parser.add_argument("--env-file", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)

    sync_parser = subparsers.add_parser(
        "sync", help="Create or verify the immutable official gold dataset"
    )
    sync_parser.add_argument("--env-file", type=Path)

    experiment_parser = subparsers.add_parser(
        "experiment", help="Run and save one LangSmith experiment"
    )
    experiment_parser.add_argument("--profile", choices=tuple(EXPERIMENT_PROFILES), required=True)
    experiment_parser.add_argument("--protocol", choices=tuple(EXPERIMENT_PROTOCOLS), required=True)
    experiment_parser.add_argument("--env-file", type=Path)
    experiment_parser.add_argument("--output", type=Path, required=True)

    augment_parser = subparsers.add_parser(
        "augment", help="Generate a separate review-required robustness candidate dataset"
    )
    augment_parser.add_argument("--per-case", type=int, choices=(1, 2), required=True)
    augment_parser.add_argument("--env-file", type=Path)
    return parser


async def async_main(args: argparse.Namespace) -> int:
    if args.command == "run":
        agent_settings = load_agent_settings(args.env_file)
        profile = get_experiment_profile(args.profile)
        cases = await anyio.to_thread.run_sync(load_cases, args.suite)
        results = await run_suite(agent_settings, args.suite, profile, cases)
        await write_local_results(
            args.output,
            suite=args.suite,
            profile=profile,
            cases=cases,
            results=results,
        )
        print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
        return 0 if all(result.passed for result in results) else 1

    cases = await anyio.to_thread.run_sync(load_cases, "full")
    if args.command == "sync":
        langsmith_settings = load_langsmith_settings(args.env_file)
        client = create_langsmith_client(langsmith_settings)
        sync_summary = await anyio.to_thread.run_sync(sync_official_dataset, client, cases)
        print(json.dumps(sync_summary, indent=2))
        return 0
    if args.command == "experiment":
        evaluation_settings = load_evaluation_settings(args.env_file)
        client = create_langsmith_client(evaluation_settings)
        profile = get_experiment_profile(args.profile)
        protocol = get_experiment_protocol(args.protocol)
        experiment_summary = await run_official_benchmark(
            client=client,
            settings=evaluation_settings,
            cases=cases,
            profile=profile,
            protocol=protocol,
            output_path=args.output,
        )
        print(json.dumps(experiment_summary, indent=2, default=str))
        return 0
    if args.command == "augment":
        augmentation_settings = load_augmentation_settings(args.env_file)
        client = create_langsmith_client(augmentation_settings)
        augmentation_summary = await generate_augmentation_candidates(
            client=client,
            settings=augmentation_settings,
            seeds=cases,
            candidates_per_case=args.per_case,
        )
        print(json.dumps(augmentation_summary, indent=2, default=str))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
