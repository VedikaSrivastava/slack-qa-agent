# Observability and evaluation

## Decision

LangSmith was considered first because it integrates directly with LangChain tracing and
experiments. It was not included in the take-home runtime because hosted tracing sends prompts and
outputs outside the local environment, while self-hosting LangSmith requires more infrastructure
than this project needs.

The local environment instead uses Langfuse as an optional trace UI. Its Docker Compose
`observability` profile keeps traces, spans, model calls, retrieval steps, metadata, latency, and
token usage on the developer machine. MinIO must contain a `langfuse` bucket before the OTEL
ingestion path can persist events; the compose MinIO service creates that directory on startup so
fresh and existing volumes do not 500 every export.

Evaluation remains separate from observability. The deterministic evaluator writes sanitized JSON
reports to `evals/reports/`, where reviewed, authorized evidence can be tracked and unaccepted
artifacts can remain clearly labeled for audit. This makes results reproducible and reviewable
without requiring a running trace service. The offline harness deliberately disables Langfuse even
when its keys are present in the selected environment file, so benchmark questions, retrieved
evidence, and generated answers are not duplicated into tracing. Reports include per-case
deterministic contract checks and aggregate evidence, latency, action, token, and estimated-cost
metrics. A normal `run` report marked `semantic_quality: not_judged` does not establish answer
correctness. Its evidence contract checks source attribution, citation membership/integrity, and
answerability behavior, not claim-level entailment; semantic judging and manual review remain
separate.

The final p19/r12/e13 take-home snapshot is deterministic and remains
`semantic_quality: not_judged`; its semantic interpretation comes from manual review. Existing
`candidate-*` judge artifacts are unaccepted because they predate recorded transfer
acknowledgement, and no Langfuse trace output is used as evaluation evidence.

## Production approach

For production, use a managed observability service or a properly operated self-hosted deployment
instead of the local Docker stack. A hosted service is a data-transfer boundary because traces may
contain prompts, retrieved evidence, model/tool inputs, and answers. The production setup should
therefore require explicit approval plus access controls, prompt and evidence redaction, retention
limits, trace sampling, cost and latency alerts, durable evaluation datasets, and separate
production and development projects. Versioned JSON evaluation can serve as deterministic
regression evidence in an authorized offline or CI workflow, while production traces support
debugging and operational monitoring.

This separation keeps the roles explicit: observability supports interactive diagnosis of agent
executions, while evaluation provides versioned evidence of behavior and regressions.
