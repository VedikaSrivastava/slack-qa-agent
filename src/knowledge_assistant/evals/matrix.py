"""Run one dataset across many profiles and prompt variants, with repeats, into one report.

This is the deterministic screening loop. For a fixed dataset and prompt/retrieval version it
compares lexical/source diagnostics, runtime contracts, actions, cost, latency, and run-to-run
variance. It does not claim semantic answer accuracy; finalists require the reference-based judge.

Output layout (under ``--output-dir``, default ``evals/reports/<label>/``):

    <profile>.json     one profile's pooled + per-repeat metrics (written as soon as it finishes)
    rollup.json        every profile side by side, plus the run manifest
    README.md          leaderboard tables for a quick read

Each ``<profile>.json`` is self-contained and re-run-safe: ``--resume`` skips a compatible profile
whose file already exists, so a long matrix survives an interruption. Starting without
``--resume`` fails if the label directory already contains output.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import uuid
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, NotRequired, TypedDict, cast

import anyio

from knowledge_assistant.agent.profiles import (
    MODEL_MATRIX_PROFILES,
    RETRIEVAL_TUNING_PROFILES,
    AgentProfile,
    get_experiment_profile,
)
from knowledge_assistant.config import (
    APPLICATION_VERSION,
    PROMPT_VERSION,
    RETRIEVAL_VERSION,
)
from knowledge_assistant.evals._stats import mean as _mean
from knowledge_assistant.evals.datasets import (
    EVALUATION_PROTOCOL_VERSION,
    SUITE_CHOICES,
    annotation_digest,
    dataset_digest,
)
from knowledge_assistant.evals.graph_follow_up_routing import (
    GraphFollowUpRoutingEvalCase,
    graph_follow_up_routing_metrics,
    load_graph_follow_up_routing_cases,
    run_graph_follow_up_routing_suite,
)
from knowledge_assistant.evals.harness import load_eval_agent_settings
from knowledge_assistant.evals.metrics import suite_metrics
from knowledge_assistant.evals.models import EvalCase, EvalResult
from knowledge_assistant.evals.runner import load_cases, run_suite
from knowledge_assistant.integrations.slack.routing import ResponderPromptVariant


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be at least 1")
    return parsed


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.SubprocessError, OSError):
        return None


def _git_is_dirty(*, output_dir: Path) -> bool | None:
    """Report source-worktree changes while excluding this matrix's generated output."""

    try:
        repository_root = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        ).resolve()
        command = [
            "git",
            "-C",
            str(repository_root),
            "status",
            "--porcelain=v1",
            "--untracked-files=all",
            "--",
            ".",
        ]
        try:
            relative_output_dir = output_dir.resolve().relative_to(repository_root)
        except ValueError:
            pass
        else:
            if relative_output_dir.parts:
                output_pathspec = relative_output_dir.as_posix()
                command.extend(
                    [
                        f":(exclude){output_pathspec}",
                        f":(exclude){output_pathspec}/**",
                    ]
                )
        return bool(subprocess.check_output(command, stderr=subprocess.DEVNULL).strip())
    except (subprocess.SubprocessError, OSError):
        return None


type EvalRepeat = list[EvalResult] | None
type FollowUpRepeat = dict[str, object] | None


class EvalRunError(TypedDict):
    """Safe persisted error metadata; provider messages may contain sensitive content."""

    error_code: str
    profile: str
    repeat_number: int
    repeat_count: int
    exception_class: str
    is_transient: bool
    prompt_variant: NotRequired[str]


def _pooled_and_repeat_view(
    repeats: list[EvalRepeat],
    *,
    model_name: str,
    answer_model_name: str | None = None,
) -> dict[str, object]:
    """Pooled metrics over every repeat, plus a per-repeat and per-case reliability view."""

    completed_repeats = [repeat for repeat in repeats if repeat is not None]
    pooled = [result for repeat in completed_repeats for result in repeat]
    per_repeat_contract_rate = [
        (sum(result.strict_contract_passed for result in repeat) / len(repeat) if repeat else 0.0)
        for repeat in repeats
    ]
    first_completed_repeat = next(iter(completed_repeats), [])
    case_ids = [result.case_id for result in first_completed_repeat]
    per_case_contract_pass_count = {
        case_id: sum(
            1
            for repeat in completed_repeats
            for result in repeat
            if result.case_id == case_id and result.strict_contract_passed
        )
        for case_id in case_ids
    }
    flaky_contract_case_ids = [
        case_id
        for case_id, passes in per_case_contract_pass_count.items()
        if 0 < passes < len(repeats)
    ]
    always_contract_fail_case_ids = [
        case_id for case_id, passes in per_case_contract_pass_count.items() if passes == 0
    ]
    # Last repeat, compact, for eyeballing what failed without persisting every answer.
    last_completed_repeat = completed_repeats[-1] if completed_repeats else []
    last_repeat_cases = [
        {
            "case_id": result.case_id,
            "strict_contract_passed": result.strict_contract_passed,
            "failed_gates": [
                check.name for check in result.checks if check.is_gate and not check.passed
            ],
            "lexical_hits": {
                group: [hit.matched, hit.total] for group, hit in result.lexical_hits.items()
            },
            "answer_words": result.answer_words,
            "tool_calls": result.tool_call_count,
            "model_calls": result.model_call_count,
            "retrieval_rounds": result.retrieval_round_count,
        }
        for result in last_completed_repeat
    ]
    return {
        "pooled": suite_metrics(pooled, model_name=model_name, answer_model_name=answer_model_name),
        "attempted_repeats": len(repeats),
        "completed_repeats": len(completed_repeats),
        "repeat_completion_rate": len(completed_repeats) / len(repeats) if repeats else 0.0,
        "per_repeat_strict_contract_rate": per_repeat_contract_rate,
        "strict_contract_rate_mean": _mean(per_repeat_contract_rate),
        "strict_contract_rate_min": (
            min(per_repeat_contract_rate) if per_repeat_contract_rate else None
        ),
        "strict_contract_rate_max": (
            max(per_repeat_contract_rate) if per_repeat_contract_rate else None
        ),
        "per_case_contract_pass_count": per_case_contract_pass_count,
        "flaky_contract_case_ids": flaky_contract_case_ids,
        "always_contract_fail_case_ids": always_contract_fail_case_ids,
        "last_repeat_cases": last_repeat_cases,
    }


_TRANSIENT_ERROR_MARKERS = (
    "connection error",
    "apiconnectionerror",
    "timeout",
    "temporarily unavailable",
    "bad record mac",
    "rate limit",
    "429",
    "502",
    "503",
    "504",
)


def _looks_transient(exc: BaseException) -> bool:
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(marker in text for marker in _TRANSIENT_ERROR_MARKERS)


def _sanitized_error(
    *,
    error_code: str,
    profile: AgentProfile,
    repeat_index: int,
    repeat_count: int,
    exc: BaseException,
    prompt_variant: ResponderPromptVariant | None = None,
) -> EvalRunError:
    """Return stable failure context without persisting raw provider or model output."""

    error = EvalRunError(
        error_code=error_code,
        profile=profile.name,
        repeat_number=repeat_index + 1,
        repeat_count=repeat_count,
        exception_class=type(exc).__name__,
        is_transient=_looks_transient(exc),
    )
    if prompt_variant is not None:
        error["prompt_variant"] = prompt_variant.value
    return error


async def _run_profile_suite(
    settings: object,
    suite: str,
    profile: AgentProfile,
    cases: list[EvalCase],
    repeats: int,
) -> tuple[list[EvalRepeat], list[EvalRunError]]:
    all_repeats: list[EvalRepeat] = []
    errors: list[EvalRunError] = []
    for repeat_index in range(repeats):
        # One deterministic attempt plus two retries, but only for transient network errors;
        # a schema/validation failure (e.g. a weak model breaking `RetrievalPlan`) is recorded
        # immediately rather than retried into the same wall.
        for attempt in range(3):
            try:
                results = await run_suite(settings, suite, profile, cases)  # type: ignore[arg-type]
                all_repeats.append(results)
                break
            except Exception as exc:
                if _looks_transient(exc) and attempt < 2:
                    print(
                        f"[matrix] {profile.name} repeat {repeat_index + 1} transient "
                        f"{type(exc).__name__}, retrying ({attempt + 1}/2)",
                        flush=True,
                    )
                    await anyio.sleep(5 * (attempt + 1))
                    continue
                error = _sanitized_error(
                    error_code="matrix_suite_repeat_failed",
                    profile=profile,
                    repeat_index=repeat_index,
                    repeat_count=repeats,
                    exc=exc,
                )
                errors.append(error)
                print(
                    f"[matrix] {profile.name} repeat {repeat_index + 1}/{repeats} failed "
                    f"({error['exception_class']}); provider details omitted",
                    flush=True,
                )
                all_repeats.append(None)
                break
    return all_repeats, errors


async def _run_profile_follow_up(
    settings: object,
    profile: AgentProfile,
    cases: list[GraphFollowUpRoutingEvalCase],
    variants: list[ResponderPromptVariant],
    repeats: int,
) -> dict[str, object]:
    per_variant: dict[str, object] = {}
    for variant in variants:
        repeat_metrics: list[FollowUpRepeat] = []
        errors: list[EvalRunError] = []
        for repeat_index in range(repeats):
            try:
                results = await run_graph_follow_up_routing_suite(
                    settings,  # type: ignore[arg-type]
                    profile,
                    cases,
                    variant,
                )
                repeat_metrics.append(graph_follow_up_routing_metrics(results))
            except Exception as exc:
                error = _sanitized_error(
                    error_code="matrix_follow_up_repeat_failed",
                    profile=profile,
                    repeat_index=repeat_index,
                    repeat_count=repeats,
                    exc=exc,
                    prompt_variant=variant,
                )
                errors.append(error)
                print(
                    f"[matrix] {profile.name}/{variant.value} repeat "
                    f"{repeat_index + 1}/{repeats} failed ({error['exception_class']}); "
                    "provider details omitted",
                    flush=True,
                )
                repeat_metrics.append(None)
        per_variant[variant.value] = _follow_up_repeat_view(repeat_metrics, errors=errors)
    return per_variant


def _follow_up_repeat_view(
    repeat_metrics: list[FollowUpRepeat],
    *,
    errors: list[EvalRunError],
) -> dict[str, object]:
    """Aggregate follow-up metrics using the current strict-contract field names."""

    completed_metrics = [metric for metric in repeat_metrics if metric is not None]
    return {
        "attempted_repeats": len(repeat_metrics),
        "completed_repeats": len(completed_metrics),
        "repeat_completion_rate": (
            len(completed_metrics) / len(repeat_metrics) if repeat_metrics else 0.0
        ),
        "routing_action_accuracy_mean": _mean(
            [float(_dig(metric, "routing", "action_accuracy") or 0.0) for metric in repeat_metrics]
        ),
        "respond_precision_mean": _mean(
            [
                float(_dig(metric, "routing", "respond_precision") or 0.0)
                for metric in repeat_metrics
            ]
        ),
        "respond_recall_mean": _mean(
            [float(_dig(metric, "routing", "respond_recall") or 0.0) for metric in repeat_metrics]
        ),
        "respond_f1_mean": _mean(
            [float(_dig(metric, "routing", "respond_f1") or 0.0) for metric in repeat_metrics]
        ),
        "unwanted_interruptions_total": sum(
            int(_dig(metric, "routing", "unwanted_interruptions", "count") or 0)
            for metric in completed_metrics
        ),
        "missed_follow_ups_total": sum(
            int(_dig(metric, "routing", "missed_follow_ups", "count") or 0)
            for metric in completed_metrics
        ),
        "accepted_follow_up_strict_contract_rate_mean": _mean(
            [
                float(_dig(metric, "graph", "accepted_follow_up_strict_contract_rate") or 0.0)
                for metric in repeat_metrics
            ]
        ),
        "per_repeat": repeat_metrics,
        "errors": errors,
    }


def _dig(mapping: object, *keys: str) -> Any:
    """Walk a nested metrics dict by key, returning None on any missing/non-dict hop."""

    current: object = mapping
    for key in keys:
        if not isinstance(current, dict):
            return None
        current = current.get(key)
    return current


async def run_matrix(args: argparse.Namespace) -> int:
    settings = load_eval_agent_settings(args.env_file)
    profiles = _selected_profiles(args)
    variants = [ResponderPromptVariant(value) for value in args.follow_up_variants]
    suite = args.suite
    cases = await anyio.to_thread.run_sync(load_cases, suite)
    digest = dataset_digest(cases)
    annotations = annotation_digest(cases)
    follow_up_cases = (
        await anyio.to_thread.run_sync(load_graph_follow_up_routing_cases) if variants else []
    )
    follow_up_digest = dataset_digest(follow_up_cases) if variants else None
    follow_up_annotations = annotation_digest(follow_up_cases) if variants else None

    output_dir = Path(args.output_dir) / args.label
    has_existing_output = output_dir.exists() and any(output_dir.iterdir())
    if has_existing_output and not args.resume:
        raise FileExistsError(
            f"Evaluation label already has reports: {output_dir}. Use a new label or --resume."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    git_commit = _git_commit()
    git_dirty = _git_is_dirty(output_dir=output_dir)
    expected_manifest: dict[str, object] = {
        "label": args.label,
        "suite": suite,
        "dataset_digest": digest,
        "annotation_digest": annotations,
        "case_count": len(cases),
        "repeats": args.repeats,
        "follow_up_variants": [variant.value for variant in variants],
        "follow_up_case_count": len(follow_up_cases),
        "follow_up_dataset_digest": follow_up_digest,
        "follow_up_annotation_digest": follow_up_annotations,
        "application_version": APPLICATION_VERSION,
        "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retrieval_version": RETRIEVAL_VERSION,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "python": sys.version.split()[0],
        "started_at": datetime.now(UTC).isoformat(),
        "profiles": [profile.name for profile in profiles],
    }
    manifest_path = output_dir / "manifest.json"
    manifest = expected_manifest
    if args.resume and has_existing_output:
        if not manifest_path.is_file():
            raise ValueError(
                "Existing matrix output has no manifest. Preserve it and use a new label."
            )
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                "Existing matrix manifest is unreadable. Preserve it and use a new label."
            ) from exc
        if not _can_resume_manifest(existing_manifest, expected=expected_manifest):
            raise ValueError(
                "Existing matrix manifest is incompatible with this evaluation contract. "
                "Preserve it and use a new label."
            )
        manifest = cast(dict[str, object], existing_manifest)
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[matrix] {args.label}: {len(profiles)} profiles x {args.repeats} repeats "
        f"on suite '{suite}' ({len(cases)} cases) -> {output_dir}"
    )

    profile_reports: dict[str, dict[str, object]] = {}
    for profile in profiles:
        profile_path = output_dir / f"{profile.name}.json"
        if args.resume and profile_path.is_file():
            try:
                existing_report = json.loads(profile_path.read_text("utf-8"))
            except (OSError, json.JSONDecodeError):
                existing_report = None
            else:
                if _can_resume_report(
                    existing_report,
                    profile=profile,
                    suite=suite,
                    dataset_digest_value=digest,
                    annotation_digest_value=annotations,
                    repeats=args.repeats,
                    follow_up_variants=[variant.value for variant in variants],
                    follow_up_case_count_value=len(follow_up_cases),
                    follow_up_dataset_digest_value=follow_up_digest,
                    follow_up_annotation_digest_value=follow_up_annotations,
                    git_commit_value=git_commit,
                    git_dirty_value=git_dirty,
                ):
                    print(f"[matrix] resume: reusing compatible report for {profile.name}")
                    profile_reports[profile.name] = existing_report
                    continue
            raise ValueError(
                f"Existing report for {profile.name} is incompatible with this evaluation "
                "contract. Preserve it and use a new label."
            )

        print(f"[matrix] {profile.name} ({profile.model_name}) ...", flush=True)
        started = datetime.now(UTC)
        repeats, suite_errors = await _run_profile_suite(
            settings, suite, profile, cases, args.repeats
        )
        follow_up: dict[str, object] = {}
        if variants and any(repeat is not None for repeat in repeats):
            follow_up = await _run_profile_follow_up(
                settings,
                profile,
                follow_up_cases,
                variants,
                args.repeats,
            )

        report: dict[str, object] = {
            "profile": asdict(profile),
            "suite": suite,
            "dataset_digest": digest,
            "annotation_digest": annotations,
            "follow_up_case_count": len(follow_up_cases),
            "follow_up_dataset_digest": follow_up_digest,
            "follow_up_annotation_digest": follow_up_annotations,
            "application_version": APPLICATION_VERSION,
            "evaluation_protocol_version": EVALUATION_PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "retrieval_version": RETRIEVAL_VERSION,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "run_id": uuid.uuid4().hex[:12],
            "started_at": started.isoformat(),
            "finished_at": datetime.now(UTC).isoformat(),
            "suite_errors": suite_errors,
            "deterministic_diagnostics": _pooled_and_repeat_view(
                repeats,
                model_name=profile.model_name,
                answer_model_name=profile.answer_model(),
            ),
            "follow_up_routing": follow_up,
        }
        profile_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        profile_reports[profile.name] = report
        _write_rollup(output_dir, manifest, profile_reports)
        _write_readme(output_dir, manifest, profile_reports)
        print(f"[matrix] {profile.name} done -> {profile_path.name}")

    _write_rollup(output_dir, manifest, profile_reports)
    _write_readme(output_dir, manifest, profile_reports)
    print(f"[matrix] complete -> {output_dir / 'README.md'}")
    return _matrix_exit_code(profile_reports)


def _matrix_exit_code(profile_reports: dict[str, dict[str, object]]) -> int:
    """Fail automation when any requested profile did not complete cleanly."""

    for report in profile_reports.values():
        if report.get("suite_errors"):
            return 1
        follow_up = report.get("follow_up_routing")
        if isinstance(follow_up, dict) and any(
            isinstance(variant_report, dict) and variant_report.get("errors")
            for variant_report in follow_up.values()
        ):
            return 1
    return 0


_MATRIX_MANIFEST_CONTRACT_FIELDS = (
    "label",
    "suite",
    "dataset_digest",
    "annotation_digest",
    "case_count",
    "repeats",
    "follow_up_variants",
    "follow_up_case_count",
    "follow_up_dataset_digest",
    "follow_up_annotation_digest",
    "application_version",
    "evaluation_protocol_version",
    "prompt_version",
    "retrieval_version",
    "git_commit",
    "git_dirty",
    "python",
    "profiles",
)


def _can_resume_manifest(manifest: object, *, expected: dict[str, object]) -> bool:
    """Resume only under the exact immutable matrix contract recorded at start."""

    return bool(
        isinstance(manifest, dict)
        and isinstance(manifest.get("started_at"), str)
        and all(
            field in manifest and manifest[field] == expected.get(field)
            for field in _MATRIX_MANIFEST_CONTRACT_FIELDS
        )
    )


def _can_resume_report(
    report: object,
    *,
    profile: AgentProfile,
    suite: str,
    dataset_digest_value: str,
    annotation_digest_value: str,
    repeats: int,
    follow_up_variants: list[str],
    follow_up_case_count_value: int,
    follow_up_dataset_digest_value: str | None,
    follow_up_annotation_digest_value: str | None,
    git_commit_value: str | None,
    git_dirty_value: bool | None,
) -> bool:
    """Reuse only reports produced by the exact current evaluation contract."""

    if not isinstance(report, dict):
        return False
    diagnostics = report.get("deterministic_diagnostics")
    follow_up = report.get("follow_up_routing")
    return bool(
        report.get("profile") == asdict(profile)
        and report.get("suite") == suite
        and report.get("dataset_digest") == dataset_digest_value
        and report.get("annotation_digest") == annotation_digest_value
        and report.get("follow_up_case_count") == follow_up_case_count_value
        and report.get("follow_up_dataset_digest") == follow_up_dataset_digest_value
        and report.get("follow_up_annotation_digest") == follow_up_annotation_digest_value
        and report.get("application_version") == APPLICATION_VERSION
        and report.get("evaluation_protocol_version") == EVALUATION_PROTOCOL_VERSION
        and report.get("prompt_version") == PROMPT_VERSION
        and report.get("retrieval_version") == RETRIEVAL_VERSION
        and report.get("git_commit") == git_commit_value
        and report.get("git_dirty") == git_dirty_value
        and not report.get("suite_errors")
        and isinstance(diagnostics, dict)
        and diagnostics.get("attempted_repeats") == repeats
        and diagnostics.get("completed_repeats") == repeats
        and isinstance(follow_up, dict)
        and list(follow_up) == follow_up_variants
        and not any(
            isinstance(variant_report, dict) and variant_report.get("errors")
            for variant_report in follow_up.values()
        )
    )


def _selected_profiles(args: argparse.Namespace) -> list[AgentProfile]:
    if args.profiles:
        return [get_experiment_profile(name.strip()) for name in args.profiles.split(",")]
    chosen: list[AgentProfile] = []
    if getattr(args, "model_matrix", False):
        chosen.extend(MODEL_MATRIX_PROFILES)
    if getattr(args, "retrieval_matrix", False):
        chosen.extend(RETRIEVAL_TUNING_PROFILES)
    if not chosen:
        chosen.extend(MODEL_MATRIX_PROFILES)
    # De-duplicate while keeping order.
    seen: set[str] = set()
    unique: list[AgentProfile] = []
    for profile in chosen:
        if profile.name not in seen:
            seen.add(profile.name)
            unique.append(profile)
    return unique


def _write_rollup(
    output_dir: Path,
    manifest: dict[str, object],
    profile_reports: dict[str, dict[str, object]],
) -> None:
    rollup = {
        "manifest": manifest,
        "generated_at": datetime.now(UTC).isoformat(),
        "profiles": profile_reports,
    }
    (output_dir / "rollup.json").write_text(json.dumps(rollup, indent=2), encoding="utf-8")


def _fmt(value: object, spec: str = "", *, none: str = "n/a") -> str:
    if value is None:
        return none
    if isinstance(value, float):
        return format(value, spec) if spec else f"{value:.3f}"
    return str(value)


def _write_readme(
    output_dir: Path,
    manifest: dict[str, object],
    profile_reports: dict[str, dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append(f"# Matrix report: {manifest['label']}")
    lines.append("")
    lines.append(
        f"Suite `{manifest['suite']}` ({manifest['case_count']} cases) x "
        f"{manifest['repeats']} repeats. "
        f"prompt `{manifest['prompt_version']}`, retrieval `{manifest['retrieval_version']}`, "
        f"commit `{manifest.get('git_commit')}`, dataset digest "
        f"`{str(manifest['dataset_digest'])[:12]}`."
    )
    lines.append("")
    lines.append(
        "`lexical macro` is candidate-authored anchor coverage averaged equally across cases. "
        "`strict contract` covers exact-value, evidence-integrity, safety, and action-budget "
        "gates. `cited/retrieved IDs` cover a candidate-curated, non-exhaustive artifact set. "
        "None is semantic answer accuracy; use the reference-based judge for that. "
        f"Cost uses list prices captured {_price_date(profile_reports)}."
    )
    lines.append("")

    header = (
        "| profile | model | lexical macro | strict contract | cited IDs | retrieved IDs | "
        "tool/case | model/case | tok/case | $/1k Q | lat p50 ms | words p50 | flaky | errors |"
    )
    sep = "|" + "|".join(["---"] * 14) + "|"
    lines.append(header)
    lines.append(sep)
    for name, report in profile_reports.items():
        profile = report.get("profile", {})
        diagnostics = report.get("deterministic_diagnostics", {})
        errors = report.get("suite_errors") or []
        if not isinstance(diagnostics, dict) or "pooled" not in diagnostics:
            lines.append(
                f"| {name} | {_dig(profile, 'model_name')} | FAILED | | | | | | | | | | | "
                f"{len(errors) if isinstance(errors, list) else '?'} |"
            )
            continue
        pooled = diagnostics["pooled"]
        classify_model = str(_dig(profile, "model_name"))
        answer_model = _dig(profile, "answer_model_name")
        model_cell = (
            f"{classify_model} → {answer_model}"
            if answer_model and answer_model != classify_model
            else classify_model
        )
        row = [
            name,
            model_cell,
            _fmt(_dig(pooled, "lexical_content_coverage", "macro_per_case"), ".2f"),
            _fmt(diagnostics.get("strict_contract_rate_mean"), ".2f"),
            _fmt(_dig(pooled, "diagnostic_source_coverage", "citation_mean"), ".2f"),
            _fmt(_dig(pooled, "diagnostic_source_coverage", "retrieval_mean"), ".2f"),
            _fmt(_dig(pooled, "tool_calls", "mean_per_case"), ".1f"),
            _fmt(_dig(pooled, "model_calls", "mean_per_case"), ".1f"),
            _fmt(_dig(pooled, "tokens", "mean_total_per_case"), ".0f"),
            _fmt(_dig(pooled, "cost_usd", "per_1k_cases"), ".2f"),
            _fmt(_dig(pooled, "latency_ms", "p50"), ".0f"),
            _fmt(_dig(pooled, "answer_length", "words_p50"), ".0f"),
            str(len(diagnostics.get("flaky_contract_case_ids", []) or [])),
            str(len(errors) if isinstance(errors, list) else 0),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    follow_up_rows: list[str] = []
    for name, report in profile_reports.items():
        follow_up = report.get("follow_up_routing")
        if not isinstance(follow_up, dict) or not follow_up:
            continue
        for variant, metrics in follow_up.items():
            if not isinstance(metrics, dict):
                continue
            follow_up_rows.append(
                "| "
                + " | ".join(
                    [
                        name,
                        variant,
                        _fmt(metrics.get("routing_action_accuracy_mean"), ".2f"),
                        _fmt(metrics.get("respond_precision_mean"), ".2f"),
                        _fmt(metrics.get("respond_recall_mean"), ".2f"),
                        _fmt(metrics.get("respond_f1_mean"), ".2f"),
                        str(metrics.get("unwanted_interruptions_total")),
                        str(metrics.get("missed_follow_ups_total")),
                        _fmt(metrics.get("accepted_follow_up_strict_contract_rate_mean"), ".2f"),
                    ]
                )
                + " |"
            )
    if follow_up_rows:
        lines.append("## Follow-up routing")
        lines.append("")
        lines.append(
            "| profile | prompt variant | action acc | respond prec | respond recall | "
            "respond F1 | unwanted interrupts | missed follow-ups | accepted strict contract |"
        )
        lines.append("|" + "|".join(["---"] * 9) + "|")
        lines.extend(follow_up_rows)
        lines.append("")

    (output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _price_date(profile_reports: dict[str, dict[str, object]]) -> str:
    for report in profile_reports.values():
        date = _dig(
            report,
            "deterministic_diagnostics",
            "pooled",
            "cost_usd",
            "prices_captured_at",
        )
        if isinstance(date, str):
            return date
    return "n/a"


def add_matrix_subcommand(subparsers: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = subparsers.add_parser(
        "matrix", help="Run one suite across many profiles with repeats into one report"
    )
    parser.add_argument("--label", default="matrix", help="Report subdirectory name")
    parser.add_argument("--suite", choices=SUITE_CHOICES, default="full")
    parser.add_argument("--repeats", type=_positive_int, default=3)
    parser.add_argument(
        "--profiles",
        default="",
        help="Comma-separated experiment profile names; overrides --model-matrix",
    )
    parser.add_argument("--model-matrix", action="store_true", help="Include the model matrix")
    parser.add_argument(
        "--retrieval-matrix",
        action="store_true",
        help=(
            "Compare global BM25 with scenario-diversified first-pass settings using one fixed "
            "model"
        ),
    )
    parser.add_argument(
        "--follow-up-variants",
        default="",
        help="Comma-separated responder prompt variants to also evaluate (e.g. current,latest_agent_context)",
    )
    parser.add_argument("--env-file", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("evals/reports"))
    parser.add_argument(
        "--resume", action="store_true", help="Skip profiles whose report file already exists"
    )


def normalize_matrix_args(args: argparse.Namespace) -> argparse.Namespace:
    if isinstance(args.follow_up_variants, str):
        args.follow_up_variants = [
            value.strip() for value in args.follow_up_variants.split(",") if value.strip()
        ]
    return args
