# Matrix report: matrix

Suite `full` (7 cases) x 3 repeats. prompt `v22`, retrieval `v12`, commit `24170ce`, dataset digest `ac16383cd1ad`.

`lexical macro` is candidate-authored anchor coverage averaged equally across cases. `strict contract` covers exact-value, evidence-integrity, safety, and action-budget gates. `cited/retrieved IDs` cover a candidate-curated, non-exhaustive artifact set. None is semantic answer accuracy; use the reference-based judge for that. Cost uses list prices captured 2026-08-29.

| profile | model | lexical macro | strict contract | cited IDs | retrieved IDs | tool/case | model/case | tok/case | $/1k Q | lat p50 ms | words p50 | flaky | errors |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| split-gpt-5.4-hybrid | gpt-5.4-nano → gpt-5.4 | 0.62 | 0.81 | 0.72 | 0.89 | 4.1 | 4.5 | 26744 | n/a | 10262 | 136 | 1 | 0 |

