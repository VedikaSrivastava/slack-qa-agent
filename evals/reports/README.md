# Evaluation reports

This directory preserves successful, failed, historical, and unaccepted evaluation artifacts.
Reports record dataset and annotation digests, application/prompt/retrieval/evaluation versions,
profile, timestamps, status, and case-level metrics. Failed reports contain sanitized error classes,
not provider bodies or credentials. New runs use unique names and are never overwritten.

## Final deterministic report

[`submission-final-v22/`](submission-final-v22/) is the current production-profile snapshot: prompt
`v22`, retrieval `v12`, evaluation protocol `v13`, and `split-gpt-5.4-hybrid`, over three repeats
(21 case-runs). `matrix/rollup.json` holds the metrics and `answers.json` holds generated answer
text from a separate single-repeat `run`.

- strict contract: 6/7 on each repeat, 18/21 pooled;
- exact content: 0.75 of 12 applicable case-runs;
- evidence contract: 0.9048 of 21;
- operations contract: 1.0 of 21;
- safety: N/A, with zero applicable cases;
- 89 tool calls, 92 model calls, and 544,947 agent tokens;
- estimated agent cost: not available from the split pricing path;
- latency: 10,376 ms p50, 20,785 ms p95, 24,391 ms max;
- `flaky_contract_case_ids`: empty; and
- `semantic_quality: not_judged`.

`official-blueharbor-defection-risk` fails all three repeats, via `exact_dates` when it answers and
`answerability_behavior` on the two case-runs where it abstains.

[`v22-flakiness-check/`](v22-flakiness-check/) is deliberately retained as the failing artifact for
a rejected first attempt at prompt v22, which scored 6/7, 6/7, 5/7 and made
`official-canada-approval-pattern` flaky. See
[Evaluation findings](../../docs/evaluation-findings.md#answer-presentation-investigation-v20-v22).

[`experiment-final-gpt54-hybrid/`](experiment-final-gpt54-hybrid/) is the prompt-`v19` snapshot of
the same profile at 6/7 strict on one repeat, superseded by the v22 report above.

[`takehome-agent-p19-r12-e13-final-20260829-01.json`](takehome-agent-p19-r12-e13-final-20260829-01.json)
is the historical `balanced-gpt-4.1-mini` snapshot at prompt `v19`: 6/7 strict, 3/4 applicable exact
content, 32 tool and 32 model calls, 191,893 tokens, $0.084898 estimated, 12,088 ms p50 and
74,796 ms p95/max.

Exit code zero means the run completed and wrote this report, not that every gate or answer passed.
The evidence contract checks source attribution, citation membership/integrity, and answerability
behavior. It does not prove claim-level entailment. Curated source IDs and lexical anchors are
candidate-authored diagnostics, not assignment gold.

Manual assignment-reference review found 5/7 fully complete, reference-agreeing answers, so the
7/7 target was not met. Prompt v19/retrieval v12 retain the p18/r11 provenance and full-evidence
safeguards, replace a keyword ranking gate with typed semantic planning, and preserve more distinct
comparison follow-up dimensions within the hard budget. They did not improve the manual outcome.
Calls, tokens, cost, and median latency fell, but one slow cohort case caused a large tail-latency
regression. Matching p19/r12 derived and multi-turn regressions were not measured.

## Experiment history

- `takehome-agent-p18-r11-e13-final-20260829-01.json` is a historical 6/7 strict snapshot.
- `takehome-agent-p17-r10-e13-final-20260829-01.json` is a historical 6/7 strict snapshot.
- `takehome-agent-p16-r9-e13-final-20260829-01.json` is a rejected 5/7 strict regression.
- `takehome-agent-p15-r8-e13-final-20260829-01.json` is a historical 6/7 strict snapshot.
- Other mixed-version completed and failed reports remain audit history, not final comparisons.

Every `candidate-*` directory is **unaccepted** and outside the documented authorization for
transferring generated answers and references to a judge model. None records authorization
metadata. The eight non-empty directories currently preserved are:

- `candidate-current-p15-r8-e13-j4-derived-20260829-01/`;
- `candidate-current-p15-r8-e13-j4-official-20260829-01/`;
- `candidate-final-p14-r8-e13-j4-derived-20260829-01/`;
- `candidate-final-p14-r8-e13-j4-multiturn-20260829-01/`;
- `candidate-final-p14-r8-e13-j4-official-20260829-01/`;
- `candidate-derived-semantic-v12-20260829-01/`;
- `candidate-multiturn-semantic-v12-20260829-01/`; and
- `candidate-official-semantic-v12-20260829-01/`.

Preserve them for audit, do not quote their rates, and do not use them as submission evidence.

Any future judge command must be run only with explicit authorization and must include
`--confirm-data-transfer`. Judge protocol v5 then records `data_transfer_acknowledged: true` and
reports combined semantic-plus-strict `task_quality_passed` diagnostics. Offline evaluation
deliberately disables Langfuse even when its keys are present.
