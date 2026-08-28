# Project documentation

The root [README](../README.md) is the complete local setup and run guide, including Slack app
configuration and end-to-end testing. Supporting implementation documentation lives here:

- [Architecture and request lifecycle](architecture.md): component boundaries, execution flow,
  health semantics, and data-ownership/retention boundaries.
- [Slack thread and Agent Session model](thread-and-session-model.md): exact conversation identity,
  follow-up, participant, ordering, clarification, and Stop semantics.
- [Engineering decisions and tradeoffs](decisions-and-tradeoffs.md): why the current technologies
  and constraints were selected.
- [Implementation journal](implementation-journal.md): detailed investigation history, including
  the two-stage Slack manifest setup blocker, Slack-derived state retention gap, Inngest readiness
  gate, rejected approaches, resolved failure modes, and remaining live validation.
- [Evaluations and LangSmith](evaluations.md): local benchmarks, hosted experiments, and dataset
  separation.

`DESIGN.md` remains separate and is intentionally not maintained by these documents.
