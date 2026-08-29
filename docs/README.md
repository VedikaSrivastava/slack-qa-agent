# Project documentation

The root [README](../README.md) is the complete local setup and run guide, including Slack app
configuration and end-to-end testing. Supporting implementation documentation lives here:

- [Architecture and request lifecycle](architecture.md): component boundaries, execution flow,
  agent response modes, per-turn provenance, action budgets, failure paths, scaling properties,
  health semantics, and data-ownership/retention boundaries.
- [Slack thread and Agent Session model](thread-and-session-model.md): exact conversation identity,
  follow-up, participant, ordering, clarification, and Stop semantics.
- [Engineering decisions and tradeoffs](decisions-and-tradeoffs.md): why the current technologies
  and constraints were selected.
- [Implementation journal](implementation-journal.md): detailed investigation history, including
  the two-stage Slack manifest setup blocker, Slack-derived state retention gap, Inngest readiness
  gate, rejected approaches, resolved failure modes, and remaining live validation.
- [Evaluations](evaluations.md): immutable and robustness datasets, retrieval screening, focused
  model comparison, graph follow-up coverage, metrics, and tracked report-preservation rules.

`DESIGN.md` remains separate and is intentionally not maintained by these documents.
