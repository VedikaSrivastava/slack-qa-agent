# Evaluation findings

## Final deterministic snapshot

The final report is
[`takehome-agent-p19-r12-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p19-r12-e13-final-20260829-01.json).
It uses `balanced-gpt-4.1-mini`, prompt `v19`, retrieval `v12`, and evaluation protocol `v13`.

| Measure | Result |
| --- | ---: |
| Strict deterministic contract | 6/7 |
| Exact content | 3/4 applicable |
| Evidence contract | 7/7 |
| Operations contract | 7/7 |
| Safety | N/A (0 applicable) |
| Tool / model calls | 32 / 32 |
| Agent tokens | 191,893 |
| Estimated agent cost | $0.084898 |
| Latency p50 / p95 / max | 12,088 ms / 74,796 ms / 74,796 ms |

The run completed successfully and wrote a valid report. That process outcome is not a quality pass,
and the report explicitly says `semantic_quality: not_judged`. The evidence contract establishes
source attribution, citation membership/integrity, and expected answerability behavior; it does not
establish claim-level entailment for every answer statement.

## Optimization objective

Changes are selected lexicographically:

1. material factual correctness, exact values, grounding, answerability, and safety;
2. every requested part, complete entity sets, and correct grouping;
3. reasonable bounded actions; then
4. latency, cost, and length only as tie-breakers.

The assignment supplies no aggregate tolerance. The seven cases are too small to establish a
production accuracy rate.

Prompt v19/retrieval v12 retain the p18/r11 provenance and full-evidence safeguards, replace a
keyword ranking gate with a typed semantic planner decision, and preserve more distinct comparison
follow-up dimensions within the existing hard action budget. This removes dependence on a fixed
ranking-keyword list rather than adding a seven-question branch; broader-suite generalization has
not been measured. The official manual outcome stayed at 5/7. Calls, tokens, cost, and median latency
fell versus p18/r11, while one 74,796 ms cohort case made tail latency much worse. Those differences
do not compensate for the two unresolved answer-quality gaps or establish that p19 is better.

## Manual official-suite review

- **BlueHarbor proof plan:** correct customer and complete proof plan, including duration,
  mapping/weighting work, top-20 A/B test, and success threshold. It is more verbose than needed.
- **Verdant rollback:** correct approved window, exact rollback command, ruleset restoration, and
  invalidation-hook replay.
- **MapleHarvest:** correct temporary mappings and workshop outputs, including alias mapping,
  migration milestones, signed schema, and `SI-SCHEMA-REG` destination.
- **Aureum:** correctly identifies `department` and `businessUnit` and Jin's hot-reloadable
  preprocessing fix, but omits both SCIM tracing and approval latency from the assignment
  reference.
- **Defection risk:** selects Pioneer Freight from explicit NoiseGuard PoC, procurement-pressure,
  and remediation-milestone evidence, while the assignment reference names BlueHarbor. Treat this
  as assignment-reference failure plus corpus/ranking ambiguity, not an invented customer. Runtime
  code must not be optimized to force BlueHarbor.
- **North America West:** returns all 12 accounts in the correct taxonomy/search and
  duplicate-action groups.
- **Canada pattern:** finds all seven accounts and the shared migration/precedence/schema pattern.
  The answer is overlong and includes caveats about details the question did not request.

Five of seven answers were fully complete and reference-agreeing. Aureum was incomplete and the
defection answer did not agree with the assignment reference, so the explicit 7/7 target was not
met. This manual review is the semantic interpretation for the take-home. It is not an LLM-judge
score and does not remove normal live-model variance.

## Experiment history

- [`takehome-agent-p18-r11-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p18-r11-e13-final-20260829-01.json)
  is a historical prompt-`v18`/retrieval-`v11` snapshot at 6/7 strict.
- [`takehome-agent-p17-r10-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p17-r10-e13-final-20260829-01.json)
  is a historical prompt-`v17`/retrieval-`v10` snapshot at 6/7 strict.
- [`takehome-agent-p16-r9-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p16-r9-e13-final-20260829-01.json)
  is a rejected prompt-`v16`/retrieval-`v9` regression at 5/7 strict.
- [`takehome-agent-p15-r8-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p15-r8-e13-final-20260829-01.json)
  is a historical prompt-`v15`/retrieval-`v8` snapshot at 6/7 strict.
- Earlier mixed-version successes and failures remain audit history and are not comparable final
  candidates.

Every `evals/reports/candidate-*` directory is unaccepted and outside the documented authorization
for sending generated answers and references to a judge model. None records authorization metadata.
Preserve these artifacts for audit, do not quote their rates, and do not use them as submission
evidence.

## Take-home versus production

For this take-home, the official deterministic run plus manual review is proportionate. Matching
p19/r12 derived and multi-turn regression runs were not measured. Those suites are
candidate-authored and would provide broader regression coverage, not independent held-out proof.

Production requires representative and held-out cases, deterministic checks where possible,
human-calibrated model judging only for semantic properties deterministic checks cannot measure,
claim-level citation entailment, repeated runs with variance estimates, CI regression checks, and
monitoring for drift, abstention, unsupported claims, errors, latency, and cost. See
[Evaluations](evaluations.md) for the full contract.

Offline evaluation deliberately disables Langfuse even when its keys are configured.
