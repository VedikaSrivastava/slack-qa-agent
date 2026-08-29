# Evaluation reset and current findings

## Status

The earlier generated reports and their conclusions were invalidated after the runtime behavior
changed materially. They predate the current report-preservation policy and are not a valid
baseline. There is currently no valid post-fix model or retrieval winner. The deterministic test
suite is green, but fresh live retrieval and model matrices are still required before selecting new
defaults.

This file records why the old results were invalidated, what has been established directly from code
and deterministic tests, and what remains to be measured. It does not treat a failed or incomplete
evaluation attempt as evidence of model quality.

## Why the earlier analysis was invalidated

The original experiments mixed model comparisons with unresolved runtime and evaluation issues:

- global BM25 could let one heavily documented scenario dominate a broad top-K result;
- invalid structured planner output could become guessed fallback behavior;
- failed matrix repetitions were not represented clearly enough in aggregate reliability;
- resume compatibility did not cover the complete evaluation contract;
- source display and prior-turn evidence reuse were not separated from answer generation;
- too many small-model and budget variants made attribution noisy before retrieval was stable.

Because those concerns affect retrieval recall, routing, action counts, and answer correctness, the
old reports cannot be compared fairly with the current graph. The reports were deleted rather than
kept as an apparent baseline.

## Post-reset implementation facts

The following are verified implementation properties, not live-model conclusions:

- the official seven-question dataset remains immutable and is not imported by runtime code;
- failed repeats count against completion and cause a non-zero matrix exit;
- `--resume` requires the same dataset digest, application/evaluation/prompt/retrieval versions,
  profile, suite, repeat count, requested follow-up variants, and an error-free completed report;
- retrieval recall, citation recall, deterministic answer checks, latency, tokens, cost, tool calls,
  model calls, and retrieval rounds are reported separately;
- global BM25 and first-pass diversification values one, two, and three are isolated with one fixed
  model and identical graph budgets;
- the model matrix contains five focused GPT-4/5.5/5.6 profiles with identical retrieval and action
  budgets;
- generated, adversarial, multi-turn, routing, and insufficient-evidence cases remain separate from
  the official benchmark.

## Evaluation order

Change one major variable at a time:

1. Run the four-profile retrieval screen once on the official suite with one fixed model.
2. Inspect failed traces and compare retrieval recall before answer correctness.
3. Confirm promising retrieval behavior on the derived and multi-turn suites.
4. Freeze the retrieval setting.
5. Run the five-profile model matrix for three repetitions on the official suite.
6. Confirm finalists on robustness and follow-up suites before changing production defaults.

Commands and the full metric contract are maintained in [Evaluations](evaluations.md).

## Fresh-run attempt

A fresh retrieval screen was attempted from the restricted coding environment. All four profiles
failed with API connection errors; the command exited non-zero and produced no quality conclusion.
An elevated retry was rejected because live evaluation sends benchmark questions, retrieved
knowledge-base evidence, and generated answers to the OpenAI API without a separate explicit data
transfer approval. That attempt predates the current preservation rule; its outcome remains recorded
here, but its temporary generated files are no longer available. New failed runs are retained as
sanitized reports.

## Current measured evidence

Local validation after the graph, retrieval, source-rendering, and failure-handling changes:

- Ruff lint: passed;
- Ruff format check: passed;
- strict mypy over `src` and `tests`: passed;
- pytest: 354 passed, with one third-party Inngest/Pydantic deprecation warning;
- supplied SQLite integration coverage is included in the full suite.

These results establish deterministic invariants and regression coverage. They do not establish
which retrieval first-pass value or model profile has the best live answer quality.

## Remaining evaluation limitations

- The official benchmark has only seven cases and cannot represent broad production behavior.
- Model aliases do not pin provider weights, so results can change without a repository diff.
- Live API access and authorization to send internal evidence are prerequisites for fresh matrices.
- One-repeat screening is only for eliminating weak retrieval settings; finalists require repeated
  confirmation.
- A lexical-first design may still miss semantically distant paraphrases. Hybrid retrieval should
  be evaluated only after fresh traces establish lexical recall as the remaining failure.
