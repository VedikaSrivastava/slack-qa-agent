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

## Latest candidate snapshot: hybrid split profile at prompt v22

The current submission report is
[`submission-final-v22`](../evals/reports/submission-final-v22/matrix/rollup.json).
It uses prompt `v22`, retrieval `v12`, evaluation protocol `v13`, and profile
`split-gpt-5.4-hybrid`, over **three repeats** of the seven official cases (21 case-runs):

| Measure | Result |
| --- | ---: |
| Strict contract, per repeat | 6/7, 6/7, 6/7 |
| Strict contract, pooled | 18/21 (0.8571) |
| Exact content | 0.75 over 12 applicable case-runs |
| Evidence contract | 0.9048 over 21 |
| Operations contract | 1.0 over 21 |
| Safety | N/A (0 applicable) |
| Tool / model calls | 89 / 92 (4.24 and 4.38 per case) |
| Retrieval rounds | 1.19 mean, 2 max |
| Tokens | 513,333 in / 31,614 out (25,950 per case) |
| Lexical macro | 0.6130 |
| Diagnostic retrieval / citation coverage | 0.8571 / 0.8095 |
| Latency p50 / p95 / max | 10,376 ms / 20,785 ms / 24,391 ms |
| Answer length p50 | 123 words |
| Budget-exceeded cases | 0 |
| Flaky contract cases | none |

`always_contract_fail_case_ids` is `official-blueharbor-defection-risk` and nothing else. That case
fails through **two different modes** across repeats, which the pooled check rates make visible:

- `exact_dates` at 0.5, when it answers but drops a committed milestone date;
- `answerability_behavior` at 0.9048, i.e. two of 21 case-runs abstained outright, both on this
  case.

Its lexical coverage is 0.0769, far below every other case. This is a retrieval/selection problem
on a case whose assignment reference the corpus also contradicts (see the two-axis rule above), not
a formatting problem.

Run command:

```bash
uv run python -m knowledge_assistant.evals matrix --profiles split-gpt-5.4-hybrid --suite full \
  --repeats 3 --env-file .env.local --output-dir evals/reports/submission-final-v22
```

`answers.json` in the same directory is a separate single-repeat `run` capturing full answer text,
because the `matrix` report intentionally stores metrics rather than generated prose.

The report is `semantic_quality: not_judged`.

### Repeats are required before accepting a prompt change

Single-repeat runs cannot distinguish a real improvement from live-model variance, and this was not
a theoretical concern. The first prompt-`v22` attempt scored 6/7 on a single repeat and looked like
a clean pass. Three repeats scored 6/7, 6/7, 5/7 and flagged `official-canada-approval-pattern` as
flaky, which is what exposed the regression described in
[Evaluation findings](evaluation-findings.md#answer-presentation-investigation-v20-v22).

The working rule is therefore:

1. a single repeat may screen an idea out, but may not accept it;
2. any candidate that will ship runs at least three repeats;
3. `flaky_contract_case_ids` must be empty, not merely small;
4. compare `per_repeat_strict_contract_rate` rather than one pooled number, since one bad repeat is
   averaged away by pooling;
5. inspect the failing trace before changing anything.

Three repeats of seven cases is still a small sample. It detects gross instability, not a reliable
production accuracy rate.

### Prompt version history on this profile

| Prompt | Change | Outcome |
| --- | --- | --- |
| `v19` | Typed semantic planner decision replacing a keyword ranking gate | 6/7 strict, 1 repeat |
| `v20` | Backticked commands, full milestone date chains, comparison disambiguation when several entities share a keyword | Kept 6/7 |
| `v21` | Planner rule so continue/retry/resume uses the earlier question from labelled thread context | Kept 6/7; fixes a Slack behavior not covered by the official suite |
| `v22` (first attempt) | Presentation rules plus a verifier gate rejecting drafts that expose retrieval internals | 6/7, 6/7, 5/7; Canada case became flaky |
| `v22` (shipped) | Same presentation rules, verifier gate removed | 6/7, 6/7, 6/7; zero flaky |

Only the presentation layer changed between v19 and v22. Retrieval stayed at `v12` throughout, so
the unchanged `official-blueharbor-defection-risk` failure is expected: nothing in these prompt
revisions targeted its retrieval gap.

### Controlled budget-only follow-up experiment (not a new baseline)

A separate profile keeps this exact snapshot setup and changes only:

- `max_retrieval_rounds: 3`
- `max_tool_calls: 10`

Profile:
`split-gpt-5.4-hybrid-budget3-tools10` (`experiment-budget3-hybrid-20260830-initial/official3.json`).

- `official-blueharbor-defection-risk`: strict ❌ (insufficient-evidence, `exact_dates` failed), retrieval
  rounds `3`, tool calls `7`, latency `33,558 ms`.
- `official-canada-approval-pattern`: strict ❌ (`required_customer_recall` failed), retrieval rounds `3`,
  tool calls `7`, latency `27,884 ms`.
- `official-na-west-account-groups`: strict ✅, retrieval rounds `1`, tool calls `5`, latency `16,556 ms`.

Strict on this 3-case slice: `1/3`.

This was intentionally aborted before any full-suite rerun because the controlled prefix requirement
("if first 3 are all good, then run 7") was not met.

## Historical deterministic snapshot

The earlier take-home candidate report is
[`takehome-agent-p19-r12-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p19-r12-e13-final-20260829-01.json).
It records prompt `v19`, retrieval `v12`, evaluation protocol `v13`, and profile
`balanced-gpt-4.1-mini`:

- strict contract: 6/7;
- exact content: 3/4 applicable cases;
- evidence contract: 7/7;
- operation contract: 7/7;
- safety: not applicable, with zero declared safety cases;
- 32 tool calls and 32 model calls;
- 191,893 total agent tokens and estimated agent cost of $0.084898; and
- latency of 12,088 ms p50 and 74,796 ms p95/max.

The report is `semantic_quality: not_judged`. Interpret it with the manual case review in
[Evaluation findings](evaluation-findings.md). Manual assignment-reference review found 5/7 fully
complete, reference-agreeing answers, so the 7/7 target was not met.

Prompt v19/retrieval v12 retain search-excerpt, customer, and scenario provenance in the grading and
answer stages, require full evidence before ranked selection, replace a keyword ranking gate with a
typed semantic planner decision, and preserve more comparison follow-up dimensions within the hard
action budget. This removes dependence on a fixed ranking-keyword list, but no matching broader
suite has measured generalization and official reference agreement did not improve. Calls, tokens,
cost, and median latency fell versus p18/r11, while a single slow cohort case caused a large
tail-latency regression. Correctness and completeness remain the selection objective; these
run-to-run differences are secondary diagnostics and do not establish that p19 is better.

The prompt-`v18`/retrieval-`v11`, prompt-`v17`/retrieval-`v10`, and
prompt-`v15`/retrieval-`v8` runs remain historical 6/7 strict evidence. The
prompt-`v16`/retrieval-`v9` run was rejected after a 5/7 strict regression.

## Commands and authorization

The accepted evidence for a shipping change is a three-repeat matrix on the production profile plus
manual review. Use `run` when the generated answer text itself is the artifact you need:

```bash
uv run python -m knowledge_assistant.evals matrix --profiles split-gpt-5.4-hybrid --suite full \
  --repeats 3 --env-file .env.local --output-dir evals/reports/<label>
uv run python -m knowledge_assistant.evals run --suite full \
  --profile split-gpt-5.4-hybrid --env-file .env.local \
  --output evals/reports/<label>/answers.json
```

`run` refuses to overwrite an existing report path, so a rerun needs a new filename. That is
deliberate: it prevents an experiment from silently replacing the artifact it is being compared
against.

No matching p22/r12 derived or multi-turn regression run has been measured. These candidate-authored
suites are the next broader checks, not hidden held-out proof:

```bash
uv run python -m knowledge_assistant.evals run --suite derived \
  --profile split-gpt-5.4-hybrid --env-file .env.local \
  --output evals/reports/derived-split-gpt-5.4-hybrid.json
uv run python -m knowledge_assistant.evals run --suite multiturn \
  --profile split-gpt-5.4-hybrid --env-file .env.local \
  --output evals/reports/multiturn-split-gpt-5.4-hybrid.json
```

Use a model judge only for semantic properties that deterministic checks cannot assess, and only
after explicit authorization for sending generated answers and references:

```bash
uv run python -m knowledge_assistant.evals judge --label authorized-finalists --suite full \
  --profiles split-gpt-5.4-hybrid --judge-model gpt-5 \
  --env-file .env.local --confirm-data-transfer
```

Judge protocol v5 records `data_transfer_acknowledged: true` only after that CLI confirmation. It
also reports `task_quality_passed` per case and `task_quality_pass_rate` in aggregate, requiring both
the candidate-defined semantic answer-quality gate and the strict deterministic contract. Those
fields are future authorized diagnostics, not assignment-defined thresholds.

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
