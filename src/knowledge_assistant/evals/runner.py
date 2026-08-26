"""Strict local and LangSmith evaluation CLI."""

from __future__ import annotations

import argparse
import asyncio
import json
import uuid
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

from langsmith import Client
from pydantic import TypeAdapter

from knowledge_assistant.agent.processor import create_question_processor
from knowledge_assistant.agent.profiles import (
    EXPERIMENT_PROFILES,
    AgentProfile,
    get_experiment_profile,
)
from knowledge_assistant.config import AgentRuntimeSettings, EvaluationSettings
from knowledge_assistant.evals.augmentation import generate_augmentation_candidates
from knowledge_assistant.evals.evaluators import evaluate_response
from knowledge_assistant.evals.langsmith_dataset import sync_official_dataset
from knowledge_assistant.evals.langsmith_experiment import run_official_benchmark
from knowledge_assistant.evals.models import EvalCase, EvalResult
from knowledge_assistant.evals.protocols import EXPERIMENT_PROTOCOLS, get_experiment_protocol

CASES_DIR = Path(__file__).with_name("cases")


def load_cases(suite: str) -> list[EvalCase]:
    path = CASES_DIR / f"{suite}.json"
    return TypeAdapter(list[EvalCase]).validate_json(path.read_text(encoding="utf-8"))


def _validate_env_file(env_file: Path | None) -> None:
    if env_file is not None and not env_file.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {env_file}")


def load_agent_settings(env_file: Path | None) -> AgentRuntimeSettings:
    _validate_env_file(env_file)
    return AgentRuntimeSettings(_env_file=env_file)


def load_evaluation_settings(env_file: Path | None) -> EvaluationSettings:
    _validate_env_file(env_file)
    return EvaluationSettings(_env_file=env_file)


def create_langsmith_client(settings: EvaluationSettings) -> Client:
    return Client(api_key=settings.langsmith_api_key.get_secret_value())


async def run_suite(
    settings: AgentRuntimeSettings, suite: str, profile: AgentProfile
) -> list[EvalResult]:
    cases = load_cases(suite)
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
            response = await processor.answer(
                question=case.question,
                conversation_id=conversation_id,
                agent_run_id=str(uuid.uuid4()),
            )
            results.append(evaluate_response(case, response))
    return results


def write_local_results(
    output_path: Path, *, suite: str, profile: AgentProfile, results: list[EvalResult]
) -> None:
    payload = {
        "suite": suite,
        "profile": asdict(profile),
        "passed": all(result.passed for result in results),
        "case_count": len(results),
        "results": [result.model_dump(mode="json") for result in results],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m knowledge_assistant.evals")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run deterministic local evaluation")
    run_parser.add_argument("--suite", choices=("smoke", "full"), required=True)
    run_parser.add_argument("--profile", choices=tuple(EXPERIMENT_PROFILES), required=True)
    run_parser.add_argument("--env-file", type=Path)
    run_parser.add_argument("--output", type=Path, required=True)

    sync_parser = subparsers.add_parser("sync", help="Upsert the official gold dataset")
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
        settings = load_agent_settings(args.env_file)
        profile = get_experiment_profile(args.profile)
        results = await run_suite(settings, args.suite, profile)
        write_local_results(args.output, suite=args.suite, profile=profile, results=results)
        print(json.dumps([result.model_dump(mode="json") for result in results], indent=2))
        return 0 if all(result.passed for result in results) else 1

    settings = load_evaluation_settings(args.env_file)
    client = create_langsmith_client(settings)
    cases = load_cases("full")
    if args.command == "sync":
        print(json.dumps(sync_official_dataset(client, cases), indent=2))
        return 0
    if args.command == "experiment":
        profile = get_experiment_profile(args.profile)
        protocol = get_experiment_protocol(args.protocol)
        summary = await run_official_benchmark(
            client=client,
            settings=settings,
            cases=cases,
            profile=profile,
            protocol=protocol,
            output_path=args.output,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0
    if args.command == "augment":
        summary = await generate_augmentation_candidates(
            client=client,
            settings=settings,
            seeds=cases,
            candidates_per_case=args.per_case,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0
    raise AssertionError(f"Unhandled command: {args.command}")


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return asyncio.run(async_main(args))


if __name__ == "__main__":
    raise SystemExit(main())
