# Evaluation findings

## Latest candidate deterministic snapshot (appendix)

Latest append-only artifact:
[`submission-final-v22`](../evals/reports/submission-final-v22/matrix/rollup.json).
It uses `split-gpt-5.4-hybrid`, prompt `v22`, retrieval `v12`, and evaluation protocol `v13`,
over three repeats (21 case-runs).

| Measure | Result |
| --- | ---: |
| Strict contract per repeat | 6/7, 6/7, 6/7 |
| Strict contract pooled | 18/21 (0.8571) |
| Exact content | 0.75 of 12 applicable |
| Evidence contract | 0.9048 of 21 |
| Operations contract | 1.0 of 21 |
| Safety | N/A (0 applicable) |
| Tool / model calls | 89 / 92 |
| Tokens (total) | 544,947 |
| Estimated agent cost | Not available from split pricing path |
| Lexical macro | 0.6130 |
| Diagnostic retrieval coverage | 0.8571 |
| Diagnostic citation coverage | 0.8095 |
| Latency p50 / p95 / max | 10,376 ms / 20,785 ms / 24,391 ms |
| Flaky contract cases | none |

The run completed successfully and wrote a valid report. `semantic_quality: not_judged` was
reported. The only strict-fail case is `official-blueharbor-defection-risk`, failing two ways:
`exact_dates` when it answers, and `answerability_behavior` on the two of 21 case-runs where it
abstains instead. Its lexical coverage is 0.0769 against 0.54–0.92 for every other case.

No semantic judge run has yet been executed for this snapshot; use a separate authorized
`judge` run before treating semantic behavior as complete.

## Answer-presentation investigation (v20-v22)

Prompts v20 through v22 changed how a correct answer is *presented*. Retrieval stayed at `v12`, so
no measured retrieval behavior changed across this sequence.

### The observed defects

A live Slack run of the Canada approval-bypass question produced a factually correct answer with
three presentation defects:

1. it ended with a bare `[ACCOUNT_LOOKUP_COVERAGE]` marker, because the model treated the internal
   metadata block supplied in its own prompt as a citable source;
2. it closed with an unrequested paragraph about not being able to prove the case was
   "statistically rare across the full account population", describing the retrieval mechanism to
   the user;
3. it opened with a bare negation of the question's framing rather than the finding.

Only the first is a plain bug. The second traces to a specific interaction: the lookup ran with
`purpose=filter_matches`, so the coverage block reported `MATCHING_SUBSET_ONLY`, and
`GRADE_EVIDENCE` told the grader that this status "cannot prove that the remainder of a requested
partition is empty". Population coverage therefore entered `missing_parts`, and `GENERATE_ANSWER`
correctly reported it. The grader was answering a partition question the user had not asked.

Two earlier reports show the same class of leak in prose rather than bracket form, with answers
citing "the complete and authoritative account lookup metadata", so this was a recurring pattern
rather than one bad generation.

### Why the first fix was wrong

The first v22 attempt added a rule to `VERIFY_GROUNDING` instructing it to mark a draft invalid
when it exposed retrieval internals. On a single repeat this scored 6/7 and looked correct.

Three repeats scored 6/7, 6/7, 5/7 with `official-canada-approval-pattern` newly flaky, and a
separate `run` on the same code abstained on that case entirely, returning *"I couldn't produce an
answer that was fully supported by the knowledge base."*

The graph explains the mechanism. Verification has exactly one escalation path:

```text
generate_answer -> verify_grounding -> repair_answer -> verify_repair -> reject_ungrounded_answer
```

There is no third chance and no partial-credit branch. Any rule that can invalidate a draft twice
destroys the entire answer, regardless of whether the objection was about grounding or wording. A
presentation preference had been given the authority to abstain.

### The corrected ownership split

The rule was removed and each concern given the owner whose failure mode matches its severity:

| Concern | Owner | Failure mode |
| --- | --- | --- |
| Bracketed internal block label reaching a user | `hide_internal_markers` in `agent/citations.py` | Deterministic removal; cannot fail |
| Prose that narrates retrieval mechanics | `GENERATE_ANSWER` instruction | Model may ignore it; answer survives |
| Unrequested population caveat | `GRADE_EVIDENCE` narrowing of `MATCHING_SUBSET_ONLY` | Caveat reappears; answer survives |
| Unsupported or miscited claim | `VERIFY_GROUNDING` | Repair, then abstain |

`VERIFY_GROUNDING` now states explicitly that phrasing exposing retrieval internals is a
presentation defect and must never invalidate a draft on that basis alone, so the rule cannot be
reintroduced accidentally.

`hide_internal_markers` extends the existing citation-hiding pass rather than adding a second
sanitizer. It removes a bracketed span when the span contains an `art_` identifier or when its
entire contents name a supplied prompt block (`ACCOUNT_LOOKUP_COVERAGE`,
`PLANNED_COMPARISON_FOLLOW_UP_QUERIES`). Ordinary bracketed prose such as `[documentation]`
survives, which a regression test asserts.

### Measured outcome

| Variant | Per-repeat strict | Canada case | Flaky |
| --- | --- | ---: | ---: |
| v22 with verifier gate | 6/7, 6/7, 5/7 | 2/3 | 1 |
| v22 shipped | 6/7, 6/7, 6/7 | 3/3 | 0 |

Artifacts: [`v22-flakiness-check`](../evals/reports/v22-flakiness-check/matrix/rollup.json) records
the regression and [`submission-final-v22`](../evals/reports/submission-final-v22/matrix/rollup.json)
records the corrected behavior. The failing artifact is deliberately retained.

### Transferable conclusions

- A single repeat cannot accept a prompt change. This regression passed its first run.
- Match a check's authority to its severity. A gate whose only escalation is total abstention must
  hold only claims worth abstaining over.
- Prefer deterministic sanitization for anything with an exact textual signature. The bracket leak
  needed a regex, not a judgment.
- Sanitization could not have covered the prose form of the same leak, so the prompt rule is still
  required. The two layers address different halves of one defect.

### Residual presentation gap

The shipped Canada answer names the precedence conflict, `canada.v2` versus `global.v3`, and the
`2026-03-12` migration date, but describes the remaining mechanisms generally as "a
rule-evaluation/normalization change". Its `lexical_fact_anchors` coverage is 0.5833, so anchors
such as stale caches, field alias mismatch, and delayed schema propagation are still not surfacing.
The case passes every gate; it is not a verbatim match to the reference wording.

## Controlled budget-only follow-up experiment (2026-08-30)

Profile `split-gpt-5.4-hybrid-budget3-tools10` keeps exact `experiment-final-gpt54-hybrid`
graph, model, prompt, ranking, and filtering behavior and changes only:

- `max_retrieval_rounds: 2 -> 3`
- `max_tool_calls: 8 -> 10`

The controlled first-pass run used only the following official questions:

1. `official-blueharbor-defection-risk`
2. `official-canada-approval-pattern`
3. `official-na-west-account-groups`

Report artifact:
[`official3.json`](../evals/reports/experiment-budget3-hybrid-20260830-initial/official3.json)

| Case | strict | final answer (short) | retrieval rounds | tool calls | `exact_dates` | `required_customer_recall` | retrieved artifact IDs | input/output tokens | latency |
| --- | ---: | --- | ---: | ---: | --- | --- | --- | ---: | ---: |
| official-blueharbor-defection-risk | ❌ | insufficient-evidence reply only | 3 (hit ceiling) | 7 | ❌ | n/a | 12 (including `art_ac8afb52f2d3` ... `art_7373b64fdfdd`) | 46,983 / 3,587 | 33,558 ms |
| official-canada-approval-pattern | ❌ | 2026-03-12 Canada approvals pattern (incomplete exhaustive list) | 3 (hit ceiling) | 7 | n/a | ❌ | 13 (including `art_e697b3abe158` ... `art_364eddbcbfe8`) | 34,230 / 3,015 | 27,884 ms |
| official-na-west-account-groups | ✅ | complete 12-account cohort split | 1 | 5 | n/a | ✅ | 16 (including `art_90991e25335f` ... `art_ccf5fa253d75`) | 23,645 / 2,226 | 16,556 ms |

Overall strict result on this 3-case slice: 1/3.

The run did use a third retrieval round on the first two cases, so the strict-fail
`official-blueharbor-defection-risk` was **not** limited by the `max_retrieval_rounds=2`
ceiling in this controlled slice. Because 1/3 strict did not pass, the requested full official
suite rerun was not started in this branch.

## Historical deterministic snapshot

The historical baseline report is
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

A fifth, procedural criterion was added after the v22 regression: a change is only accepted when at
least three repeats agree and `flaky_contract_case_ids` is empty. Stability is not a tie-breaker
below latency and cost. An intermittent abstention is a correctness failure on the repeats where it
happens, and pooling averages it out of view.

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
  In the historical baseline the answer was overlong and included caveats about details the
  question did not request; prompt v22 removed those caveats and shortened it from 253 to about
  150 words. See [Answer-presentation investigation](#answer-presentation-investigation-v20-v22).

Five of seven answers were fully complete and reference-agreeing in the historical baseline.
Aureum was incomplete and the defection answer did not agree with the assignment reference, so the
explicit 7/7 target was not met. This manual review is the semantic interpretation for that
historical take-home baseline. It is not an LLM-judge score and does not remove normal live-model
variance.

The prompt-v22 hybrid run has received a manual review only for the Canada presentation defects
described above. A full seven-case manual re-review against the assignment references has not been
repeated on v22, so the "5/7 fully complete and reference-agreeing" judgement remains the
historical baseline figure rather than a current measurement.

## Experiment history

- [`submission-final-v22`](../evals/reports/submission-final-v22/matrix/rollup.json) is the accepted
  prompt-`v22` snapshot on `split-gpt-5.4-hybrid`: 6/7 strict on each of three repeats, no flaky
  cases.
- [`v22-flakiness-check`](../evals/reports/v22-flakiness-check/matrix/rollup.json) is the **rejected**
  first prompt-`v22` attempt at 6/7, 6/7, 5/7 with `official-canada-approval-pattern` flaky. It is
  retained deliberately as the evidence for the verifier-gate regression.
- [`experiment-final-gpt54-hybrid`](../evals/reports/experiment-final-gpt54-hybrid/matrix/split-gpt-5.4-hybrid.json)
  is the prompt-`v19` snapshot of the same profile at 6/7 strict on one repeat, superseded above.
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
