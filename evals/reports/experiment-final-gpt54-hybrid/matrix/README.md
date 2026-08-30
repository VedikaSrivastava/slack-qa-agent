# Matrix report: matrix

Suite `full` (7 cases) x 1 repeats. prompt `v19`, retrieval `v12`, commit `24170ce`, dataset digest `ac16383cd1ad`.

`lexical macro` is candidate-authored anchor coverage averaged equally across cases. `strict contract` covers exact-value, evidence-integrity, safety, and action-budget gates. `cited/retrieved IDs` cover a candidate-curated, non-exhaustive artifact set. None is semantic answer accuracy; use the reference-based judge for that. Cost uses list prices captured 2026-08-29.

| profile | model | lexical macro | strict contract | cited IDs | retrieved IDs | tool/case | model/case | tok/case | $/1k Q | lat p50 ms | words p50 | flaky | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| split-gpt-5.4-hybrid | gpt-5.4-nano → gpt-5.4 | 0.64 | 0.86 | 0.79 | 0.86 | 3.9 | 4.3 | 24572 | n/a | 10030 | 117 | 0 | 0 |

