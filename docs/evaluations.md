# Evaluations

The evaluation commands run the same `QuestionProcessor`, LangGraph, prompts, retrieval repository,
and profile budgets used by Slack. They use an in-process checkpointer, so local evaluation needs
the bundled SQLite database and an OpenAI API key but not Slack, PostgreSQL, Inngest, or a hosted
evaluation service.

Live evaluation sends benchmark questions, retrieved knowledge-base evidence, and generated answers
to the selected OpenAI models. Treat that as an explicit data-transfer boundary and obtain the
appropriate authorization before running it.

## Dataset separation

`src/knowledge_assistant/evals/cases/full.json` is the immutable seven-question official benchmark.
Runtime code never imports it or its expected artifact IDs. Generated, adversarial, multi-turn,
prompt-injection, and insufficient-evidence cases remain in separate suites so tuning does not turn
the official examples into runtime rules.

Every matrix report records the dataset digest, application version, evaluation protocol, prompt
and retrieval versions, Git commit, complete profile, repeat count, errors, and timestamps. A
`--resume` run reuses a profile only when that full contract matches and every requested repeat
completed without an error.

## Metrics

Measure retrieval independently from answer presentation:

- retrieval recall: required artifacts appeared anywhere in bounded retrieved evidence;
- citation recall: the grounded answer cited the required artifacts internally;
- deterministic fact/entity/date/command checks and overall case pass rate;
- completed versus attempted repeats and per-case flakiness;
- tool calls, model calls, retrieval rounds, latency, tokens, and estimated cost;
- errors, which count as failed attempts and make the matrix command exit non-zero.

Artifact markers remain an internal evaluation and grounding contract. Slack hides them unless a
user requests sources.

## Retrieval screening

First hold the model and graph budgets constant and compare retrieval only:

```bash
make eval-retrieval-matrix SUITE=full REPEATS=1
```

The four profiles compare global BM25 with scenario-diversified first-pass values of one, two, and
three. Candidate over-fetch remains bounded and deferred candidates backfill in BM25 order. Inspect
failed traces before choosing a setting. Confirm a promising result on the separate derived and
multi-turn suites; do not select a default only because it wins the seven official cases.

## Focused model matrix

After retrieval is fixed, compare the five code-reviewed model profiles while holding retrieval and
action budgets constant:

```bash
make eval-matrix SUITE=full REPEATS=3
make eval-matrix SUITE=derived REPEATS=3
make eval-matrix SUITE=multiturn REPEATS=3
```

The focused set is GPT-4.1 mini, a GPT-4.1 answer-role split, GPT-5.5, GPT-5.6 Terra, and a GPT-5.6
Luna/Sol role split. It intentionally avoids a broad sweep of many small models. Compare answer
quality, retrieval/citation recall, reliability, latency, tokens, and estimated cost rather than
ranking on correctness alone.

## Follow-up routing

Responder routing has a separate suite because deciding whether an unmentioned Slack reply belongs
to the agent is different from answering a knowledge question:

```bash
make eval-followup-variants REPEATS=3
```

Track response precision and recall, unwanted interruptions, missed follow-ups, and accepted-answer
quality. Intentional turns inside one multi-turn case share a thread; independent repetitions use a
fresh LangGraph thread.

## One-profile diagnostics

Use a single run while investigating a trace or regression:

```bash
uv run python -m knowledge_assistant.evals run --suite full \
  --profile balanced-gpt-4.1-mini --env-file .env.local \
  --output evals/reports/full-balanced-gpt-4.1-mini.json
```

Reports under `evals/reports/` are tracked reviewer evidence. Preserve completed and failed runs;
use a unique label such as `post-fix-2026-08-29` when prompt, retrieval, dataset, application, or
evaluation-protocol behavior changes. Single-run commands and new matrices fail if their output
already exists. `--resume` may extend only a compatible interrupted matrix; incompatible profile
reports also fail instead of being replaced. Failed single runs record only a stable error code and
exception class, never raw provider details.
