# Evaluations

Offline evaluation invokes the same `QuestionProcessor`, LangGraph, prompts, retrieval repository,
and profile budgets used by Slack. It uses an in-process checkpointer, so PostgreSQL, Inngest,
Slack, and Langfuse are not required. The harness deliberately clears Langfuse credentials even
when they are present in the selected environment file.

Live evaluation still calls OpenAI. A deterministic `run` sends questions, retrieved evidence, and
generated answers. A semantic `judge` run additionally sends reference answers to a judge model.
Treat both as data-transfer boundaries and obtain appropriate authorization. Every judge command
must include `--confirm-data-transfer`.

## Benchmark boundaries

The seven question/reference pairs in `src/knowledge_assistant/evals/cases/full.json` are the
assignment benchmark. Runtime code does not import them or their annotations. Derived, multi-turn,
adversarial, routing, and insufficient-evidence cases stay in separate candidate-authored suites so
runtime behavior is not optimized to seven known answers.

Lexical anchors, required customer lists, and `diagnostic_source_ids` are candidate-authored
annotations. Source IDs are non-exhaustive diagnostics, not assignment gold: another retrieved
artifact may support the same answer. Required customer recall is a necessary deterministic gate
for declared exhaustive-set questions, while correct grouping and extra-name precision still need
manual or semantic review.

## Success criteria

Candidate changes are selected lexicographically, not by a weighted aggregate:

1. Material factual correctness, exact requested values, grounding, correct answerability behavior,
   and safety.
2. Completeness across every requested part, including full entity sets and correct grouping.
3. Reasonable bounded tool/model actions and retrieval rounds.
4. Latency, cost, and response length only as tie-breakers between equally correct systems.

A cheaper, faster, or shorter answer cannot compensate for a factual error or omission. The
assignment defines no overall pass threshold; the candidate target is 7/7 complete official answers
without regressions on broader suites.

The defection-risk case must be reported on two axes. Its assignment reference names BlueHarbor,
while the corpus contains explicit NoiseGuard, procurement, and remediation-milestone evidence for
Pioneer Freight. That is a reference mismatch and corpus ambiguity. Runtime code must never force
BlueHarbor or another benchmark answer.

## Deterministic contract

The normal `run` and `matrix` commands report `semantic_quality: not_judged`. Their checks include:

- exact commands and dates that the answer must supply;
- required customer recall for declared exhaustive sets;
- source attribution and citation integrity;
- expected answer versus insufficient-evidence behavior;
- tool, model, and retrieval-round budgets; and
- case-defined safety checks when applicable.

`content_exact_passed` is not applicable for cases that declare no exact content gate, so those
cases do not inflate the content-rate denominator. The evidence contract means that sources are
present when required, cited IDs belong to retrieved evidence, and answerability behavior matches
the case. It does not prove claim-level entailment between every answer statement and its citation.
Candidate lexical/source overlap remains non-gating.

`strict_contract_passed` means all applicable deterministic gates passed. It is not semantic answer
correctness. Similarly, a CLI exit code of zero means execution completed and a report was written;
it is not a quality pass.

## Final take-home snapshot

The final production-profile report is
[`takehome-agent-p17-r10-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p17-r10-e13-final-20260829-01.json).
It records prompt `v17`, retrieval `v10`, evaluation protocol `v13`, and profile
`balanced-gpt-4.1-mini`:

- strict contract: 6/7;
- exact content: 3/4 applicable cases;
- evidence contract: 7/7;
- operation contract: 7/7;
- safety: not applicable, with zero declared safety cases;
- 28 tool calls and 31 model calls;
- 173,521 total agent tokens and estimated agent cost of $0.0767848; and
- latency of 9,949 ms p50 and 19,173 ms p95/max.

The report is `semantic_quality: not_judged`. Interpret it with the manual case review in
[Evaluation findings](evaluation-findings.md). The prompt-`v16`/retrieval-`v9` run was rejected after
a 5/7 strict regression. The prompt-`v15`/retrieval-`v8` 6/7 run remains historical evidence, not
the final snapshot.

## Commands and authorization

The minimal take-home evidence is one official deterministic run plus manual review:

```bash
uv run python -m knowledge_assistant.evals run --suite full \
  --profile balanced-gpt-4.1-mini --env-file .env.local \
  --output evals/reports/full-balanced-gpt-4.1-mini.json
```

Derived and multi-turn runs are limited regression checks. They are candidate-authored and tuned,
not hidden held-out proof:

```bash
uv run python -m knowledge_assistant.evals run --suite derived \
  --profile balanced-gpt-4.1-mini --env-file .env.local \
  --output evals/reports/derived-balanced-gpt-4.1-mini.json
uv run python -m knowledge_assistant.evals run --suite multiturn \
  --profile balanced-gpt-4.1-mini --env-file .env.local \
  --output evals/reports/multiturn-balanced-gpt-4.1-mini.json
```

Use a model judge only for semantic properties that deterministic checks cannot assess, and only
after explicit authorization for sending generated answers and references:

```bash
uv run python -m knowledge_assistant.evals judge --label authorized-finalists --suite full \
  --profiles balanced-gpt-4.1-mini --judge-model gpt-5 \
  --env-file .env.local --confirm-data-transfer
```

Every `evals/reports/candidate-*` directory is unaccepted and outside the documented authorization
because it contains judge output but no authorization metadata. Preserve these directories as audit
history, do not quote their rates, and do not treat them as submission evidence. The current
inventory is listed in `evals/reports/README.md`; this rule also applies to any later `candidate-*`
artifact.

## Production evaluation

Seven official examples are a take-home benchmark, not a deployment gate. Production needs:

- representative sampled and genuinely held-out sets across question types, segments,
  answerability, and risk;
- deterministic checks wherever exact facts, commands, dates, sets, budgets, and routing can be
  assessed reliably;
- human-calibrated model judging only for semantic qualities deterministic checks cannot measure;
- claim-level citation entailment, citation precision, and unsupported-claim tracking;
- repeated paired runs and variance estimates before selecting prompts, retrieval, or models; and
- CI regression checks plus monitoring for input/retrieval drift, abstention, errors, latency, and
  cost.

Production questions or evidence must not be sent to an evaluator or hosted trace service without
an approved data-handling, retention, and deletion policy.

## Report preservation

Reports record dataset and annotation digests, application/prompt/retrieval/evaluation versions,
profile, timestamps, status, and per-case results. Preserve successful and failed experiments under
unique names. Track reviewed, authorized evidence; retain unaccepted artifacts clearly labeled for
audit. Failed reports contain sanitized error codes and exception classes, never provider bodies or
credentials.
