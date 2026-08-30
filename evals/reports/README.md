# Evaluation reports

This directory preserves successful, failed, historical, and unaccepted evaluation artifacts.
Reports record dataset and annotation digests, application/prompt/retrieval/evaluation versions,
profile, timestamps, status, and case-level metrics. Failed reports contain sanitized error classes,
not provider bodies or credentials. New runs use unique names and are never overwritten.

## Final deterministic report

[`takehome-agent-p17-r10-e13-final-20260829-01.json`](takehome-agent-p17-r10-e13-final-20260829-01.json)
is the final production-profile snapshot: prompt `v17`, retrieval `v10`, evaluation protocol `v13`,
and `balanced-gpt-4.1-mini`.

- strict contract: 6/7;
- exact content: 3/4 applicable cases;
- evidence contract: 7/7;
- operations contract: 7/7;
- safety: N/A, with zero applicable cases;
- 28 tool calls, 31 model calls, and 173,521 agent tokens;
- estimated agent cost: $0.0767848;
- latency: 9,949 ms p50 and 19,173 ms p95/max; and
- `semantic_quality: not_judged`.

Exit code zero means the run completed and wrote this report, not that every gate or answer passed.
The evidence contract checks source attribution, citation membership/integrity, and answerability
behavior. It does not prove claim-level entailment. Curated source IDs and lexical anchors are
candidate-authored diagnostics, not assignment gold.

## Experiment history

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
`--confirm-data-transfer`. Offline evaluation deliberately disables Langfuse even when its keys are
present.
