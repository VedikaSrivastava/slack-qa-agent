# Architecture and request lifecycle

## Component boundaries

```text
Slack Events API
      |
      v
exact-path ingress proxy (`/slack/events` only)
      |
      v
FastAPI + Slack Bolt -----> PostgreSQL run ledger
      |
      v
Inngest durable function
      |
      v
QuestionProcessor boundary
      |
      v
Bounded LangGraph -----> OpenAI models
      |
      +---------------> read-only SQLite knowledge database
      |                         |
      |                         +-- FTS5 lexical search
      |                         +-- structured account lookup
      v
PostgreSQL LangGraph checkpoints
      |
      v
Slack placeholder update with answer and sources
```

The `QuestionProcessor` protocol is the stable application boundary. Slack/Inngest, the diagnostic
CLI, and evaluation runners call the same processor without putting Slack types inside retrieval or
agent logic.

## Slack ingress

The public development tunnel terminates at a small proxy that forwards only
`POST /slack/events`; health, readiness, and the unsigned local Inngest development endpoint remain
off the public ingress. FastAPI mounts Slack Bolt at that path. Bolt verifies Slack signatures and
request timestamps. The app accepts `app_mention` events, ignores bot-generated messages, rejects
malformed payloads, and asks for a question when a mention contains no text.

The Slack `event_id` becomes the idempotent application event key. The handler persists a queued
run and sends an Inngest event without invoking a model in the request path. Slack receives a
successful acknowledgement only after that durable handoff succeeds, so a failed handoff is retried
with the same event identity.

## Durable execution

Inngest owns retries and side-effect ordering:

1. mark the run as started;
2. create or reuse a Slack status placeholder;
3. execute or reuse the transport-independent agent result and persist it before acknowledging the
   durable step;
4. update the placeholder with the answer;
5. persist the completed response and cited sources.

Work is serialized per Slack conversation and capped across OpenAI work. Duplicate deliveries reuse
the same run and deterministic Inngest event identity.

## Bounded answer workflow

The LangGraph workflow performs:

1. follow-up resolution using bounded conversation history;
2. retrieval planning;
3. structured account lookup and/or lexical search;
4. artifact reads within count and character budgets;
5. evidence grading;
6. at most one retrieval refinement under the production profile;
7. answer generation;
8. citation and grounding verification;
9. at most one repair followed by re-verification;
10. finalization or a clear insufficient-evidence response.

Evidence is preserved across refinement rounds, deduplicated by artifact ID, and kept within the
context budget. The graph also has an explicit recursion limit, so no path can become an open-ended
agent loop.

## Data ownership

- The supplied SQLite database is immutable knowledge input. It is opened with SQLite URI
  `mode=ro`, `immutable=1`, and `PRAGMA query_only=ON`.
- Alembic owns the application tables for the run ledger and cited sources.
- LangGraph's supported Postgres checkpointer owns its own checkpoint tables.
- Inngest owns durable function state and retry scheduling.
- LangSmith is optional observability and experiment infrastructure, not runtime source of truth.

## Health and observability

- `GET /healthz` reports process liveness without checking dependencies or exposing secrets.
- `GET /readyz` checks the knowledge file and PostgreSQL connectivity after strict configuration
  validation.
- Structured logs carry request, run, and conversation identifiers.
- PostgreSQL retains run status, model/tool/retrieval action counts, latency, model/version metadata,
  and sources.
- The Inngest UI exposes durable steps and retries.
- LangSmith tracing can be enabled explicitly for model and graph traces.
