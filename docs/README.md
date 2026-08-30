# Project documentation

The root [README](../README.md) is the complete local setup and run guide, including Slack app
configuration and end-to-end testing. Supporting implementation documentation lives here:

- [Architecture and request lifecycle](architecture.md): component boundaries, execution flow,
  agent response modes, per-turn provenance, action budgets, failure paths, scaling properties,
  health semantics, and data-ownership/retention boundaries.
- [Slack thread and Agent Session model](thread-and-session-model.md): exact conversation identity,
  follow-up, participant, ordering, clarification, Stop semantics, and recovery of a question lost
  to Stop or responder silence.
- [Engineering decisions and tradeoffs](decisions-and-tradeoffs.md): why the current technologies
  and constraints were selected, including the per-role model split and the rule that only
  grounding failures may reach the verifier.
- [Implementation journal](implementation-journal.md): detailed investigation history, including
  the two-stage Slack manifest setup blocker, Slack-derived state retention gap, Inngest readiness
  gate, native Stop hover as Slack client chrome, Stop recovery and migration `0002`, per-role
  model selection, rejected approaches, resolved failure modes, and remaining live validation.
- [Evaluations](evaluations.md): immutable and robustness datasets, retrieval screening, focused
  model comparison, graph follow-up coverage, metrics, the repeats-before-acceptance rule, the
  prompt version history, and tracked report-preservation rules.
- [Evaluation findings](evaluation-findings.md): measured snapshots, per-case manual review, and
  the answer-presentation investigation including a regression that was introduced and removed.

`DESIGN.md` remains separate and is intentionally not maintained by these documents.
