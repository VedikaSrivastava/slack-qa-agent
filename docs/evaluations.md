# Evaluations and LangSmith

The evaluation system has two independent paths:

- local evaluation writes JSON results, applies deterministic rule-based checks, and requires OpenAI
  plus the local data stores;
- LangSmith evaluation syncs a versioned dataset, runs hosted experiments, and saves reproducibility
  metadata plus aggregate statistics without returning or persisting private workspace URLs.

LangSmith is optional for the Slack runtime and local evaluator. Nothing in prompt construction or
retrieval depends on a private LangSmith resource.

## Datasets

`src/knowledge_assistant/evals/cases/full.json` is the official seven-question human-curated suite.
It includes reference answers, required facts, expected source IDs, and action budgets. Its digest
and the immutable `official-v1` LangSmith tag identify the exact evaluated content. Once that tag
exists, the sync command verifies its examples and fails on drift instead of moving the tag.

Do not mix generated cases into the official suite. Augmentation writes candidates to the separate
`slack-qa-agent-augmentation-candidates` dataset with `review_status=candidate` so they can be
reviewed before becoming a distinct robustness suite.

## Local smoke evaluation

This path does not require a LangSmith key:

```bash
docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals run --suite smoke --profile balanced-gpt-4.1-mini --output /app/evals/results/smoke-balanced-gpt-4.1-mini.json
```

Use `--suite full` to run all seven official cases. The command exits non-zero when a deterministic
case fails.

Compose writes summaries into the repository's `evals/results/` directory. Docker Desktop handles
that bind mount directly. When using native Docker Engine on Linux instead, grant the container's
non-root UID access once before running an evaluation:

```bash
setfacl -m u:10001:rwx evals/results
```

## LangSmith setup

1. Create or select a LangSmith workspace and obtain an API key.
2. Set `LANGSMITH_API_KEY` in `.env.local`.
3. Leave `LANGSMITH_TRACING=false` unless runtime Slack traces are also wanted. LangSmith experiment
   commands enable tracing for their own runs explicitly.

The `experiment` command also requires OpenAI access to the selected profile model and the
code-defined `gpt-5.6-terra` evaluator. The `augment` command uses `gpt-5.6-terra` as its generator.
The `sync` command does not invoke an OpenAI model.

The project name and dataset names are code-defined so experiments remain comparable. A reviewer
does not need access to the author's LangSmith workspace to run the repository locally; they can
use their own key and receive the same named dataset and project in their workspace.

## Sync and run an experiment

Synchronize the official dataset:

```bash
docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals sync
```

Run the screening protocol and save the summary:

```bash
docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals experiment --profile balanced-gpt-4.1-mini --protocol screening --output /app/evals/results/langsmith-balanced-gpt-4.1-mini-screening.json
```

The experiment records deterministic correctness, semantic reference correctness, retrieval recall,
citation recall, action-budget compliance, latency, token usage, estimated cost, errors, the full
agent profile, and code/dataset versions. The command fails if a repetition, target run, evaluator
result, or aggregate error rate is missing or unsuccessful. Private LangSmith URLs remain visible in
the LangSmith workspace only; the command output and result JSON omit them.

## Experiment protocol

Run `screening` once for each code-defined profile to compare model and retrieval-budget choices.
Use `confirmation` only for the strongest two profiles; it performs three repetitions per example
to expose variance. Both protocols use concurrency one for controlled rate-limit and latency
comparisons.

Available profiles are defined in `src/knowledge_assistant/agent/profiles.py`; unknown profile names
fail instead of falling back to a default.

## Generate review candidates

After the official baseline is stable:

```bash
docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals augment --per-case 2
```

The generator creates at most two paraphrase or multi-turn candidates per official seed, traces the
generation, and fails if the returned count is wrong. Human review is required before any candidate
is promoted into a separately versioned robustness suite.
