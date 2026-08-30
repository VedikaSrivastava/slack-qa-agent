# Evaluation findings

## Final deterministic snapshot

The final report is
[`takehome-agent-p17-r10-e13-final-20260829-01.json`](../evals/reports/takehome-agent-p17-r10-e13-final-20260829-01.json).
It uses `balanced-gpt-4.1-mini`, prompt `v17`, retrieval `v10`, and evaluation protocol `v13`.

| Measure | Result |
| --- | ---: |
| Strict deterministic contract | 6/7 |
| Exact content | 3/4 applicable |
| Evidence contract | 7/7 |
| Operations contract | 7/7 |
| Safety | N/A (0 applicable) |
| Tool / model calls | 28 / 31 |
| Agent tokens | 173,521 |
| Estimated agent cost | $0.0767848 |
| Latency p50 / p95 | 9,949 ms / 19,173 ms |

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

## Manual official-suite review

- **BlueHarbor proof plan:** correct customer and complete proof plan, including duration,
  mapping/weighting work, top-20 A/B test, and success threshold.
- **Verdant rollback:** correct approved window, exact rollback command, ruleset restoration, and
  invalidation-hook replay.
- **MapleHarvest:** correct temporary mappings and workshop outputs, including alias mapping,
  migration milestones, signed schema, and `SI-SCHEMA-REG` destination.
- **Aureum:** correctly identifies `department` and `businessUnit` and Jin's hot-reloadable
  preprocessing fix, but omits the SCIM tracing component in the assignment reference.
- **Defection risk:** selects Pioneer Freight from explicit NoiseGuard PoC, procurement-pressure,
  and remediation-milestone evidence, while the assignment reference names BlueHarbor. Treat this
  as assignment-reference failure plus corpus/ranking ambiguity, not an invented customer. Runtime
  code must not be optimized to force BlueHarbor.
- **North America West:** returns all 12 accounts in the correct taxonomy/search and
  duplicate-action groups.
- **Canada pattern:** finds all seven accounts and the shared migration/precedence/schema pattern.
  The answer is correct but includes long caveats about details the question did not request.

This manual review is the semantic interpretation for the take-home. It is not an LLM-judge score
and does not remove normal live-model variance.

## Experiment history

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

For this take-home, the official deterministic run plus manual review is proportionate. Derived and
multi-turn suites provide limited candidate-authored regression coverage, not independent held-out
proof.

Production requires representative and held-out cases, deterministic checks where possible,
human-calibrated model judging only for semantic properties deterministic checks cannot measure,
claim-level citation entailment, repeated runs with variance estimates, CI regression checks, and
monitoring for drift, abstention, unsupported claims, errors, latency, and cost. See
[Evaluations](evaluations.md) for the full contract.

Offline evaluation deliberately disables Langfuse even when its keys are configured.
