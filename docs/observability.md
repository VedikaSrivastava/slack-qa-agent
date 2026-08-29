# Observability and evaluation

## Decision

LangSmith was considered first because it integrates directly with LangChain tracing and
experiments. It was not included in the take-home runtime because hosted tracing sends prompts and
outputs outside the local environment, while self-hosting LangSmith requires more infrastructure
than this project needs.

The local environment instead uses Langfuse as an optional trace UI. Its Docker Compose
`observability` profile keeps traces, spans, model calls, retrieval steps, metadata, latency, and
token usage on the developer machine.

Evaluation remains separate from observability. The deterministic evaluator writes sanitized JSON
reports to the generated, ignored `evals/reports/` directory, making results reproducible and
reviewable without requiring a running
trace service. These reports include per-case checks and aggregate correctness, latency, action,
and token metrics.

## Production approach

For production, use a managed observability service or a properly operated self-hosted deployment
instead of the local Docker stack. The production setup should include access controls, prompt and
evidence redaction, retention limits, trace sampling, cost and latency alerts, durable evaluation
datasets, and separate production and development projects. Local JSON evaluation should remain in
CI as a deterministic regression gate, while production traces support debugging and operational
monitoring.

This separation keeps the roles explicit: observability supports interactive diagnosis of agent
executions, while evaluation provides versioned evidence of behavior and regressions.
