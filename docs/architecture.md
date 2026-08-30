# Architecture and request lifecycle

## Component boundaries

```text
Slack Events API
      |
      v
exact-path ingress proxy (`/slack/events` only)
      |
      v
FastAPI + Slack Bolt
      |-- app mention ------> `slack/question.received` ------+
      |-- thread reply -----> `slack/follow_up.candidate` ----+--> Inngest `route-slack-turn`
      |-- Agent Stop -------> persisted cancellation claim --> cancellation cleanup
                                                                  |
                                                                  v
                         PostgreSQL `slack_turns` causal queue + atomic `agent_runs` link
                                                                  |
                               optional classifier child <--------+
                                                                  |
                                  progress child, then question-processing child
                                                                  |
                                                   QuestionProcessor boundary
                                                                  |
                                                                  v
                                                   bounded LangGraph -----> OpenAI models
                                                                  |
                                                                  +------> read-only SQLite
                                                                  |          |-- FTS5 search
                                                                  |          +-- account lookup
                                                                  v
                                                   PostgreSQL checkpoints
                                                                  |
                                                                  v
                                             Slack native task stream + final answer
                                             (immutable final post if stream opening fails)
```

The `QuestionProcessor` protocol is the stable application boundary. Slack/Inngest, the diagnostic
CLI, and evaluation runners call the same processor without putting Slack types inside retrieval or
agent logic.

## Slack ingress

The public development tunnel terminates at a small proxy that forwards only
`POST /slack/events`; health, readiness, and the unsigned local Inngest development endpoint remain
off the public ingress. This is the signed Slack Events API Request URL. FastAPI mounts Slack Bolt
at that path. Bolt verifies Slack
signatures and request timestamps before dispatch. A single-workspace `auth.test` runs at startup;
the verified bot/workspace identity is cached so authorization adds no network call to Slack's
three-second acknowledgement path. Startup also compares the granted-scope response header with the
five manifest-required bot scopes and fails with a reinstall hint if the installed token is stale.
The app accepts `app_mention`, subscribed
channel `message`, and `agent_session_stopped` events; it ignores bot-generated messages and
irrelevant message subtypes, rejects malformed payloads, and asks for a question when a mention
contains no text. That small help response is bounded and uses a deterministic message ID so a
Slack retry does not duplicate it.

The Slack `event_id` is the immutable external event key. Parsing derives UUIDv5 run/candidate IDs
from workspace, Slack event ID, and purpose, so delivery retries reproduce the same internal
identity. Dispatch also uses stable Inngest event IDs (`slack-question:<event_id>` or
`slack-follow-up:<event_id>`). The signed request path does not create an `agent_runs` row or invoke a
model; it acknowledges only after the bounded Inngest handoff succeeds. The durable router owns
subsequent persistence and execution.

A Stop request is different because cancellation intent must win or lose against delivery before
the handoff. It first claims its Slack `event_id` in PostgreSQL, binds it to the exact reported stream
when possible, and persists whether cancellation won before emitting its cancellation event.
Replays return that immutable outcome instead of selecting a newer run. When Slack omits a usable
stream identity, the original message and Stop timestamps provide a causal upper bound; an older
delayed Stop therefore cannot select a later run.

## Multi-turn responder policy

Both explicit mentions and ordinary human replies enter `route-slack-turn`, so they share one causal
ordering path. With the default `agent_owned_thread_follow_ups` policy, an ordinary reply in a public
or private channel thread is a durable candidate. When it reaches the head, the router checks for a
delivered agent response in that exact workspace/channel/root thread, then invokes the structured,
conservative classifier as its own model-constrained Inngest function. Only `respond` creates a
question run; acknowledgements, human-to-human messages, and `uncertain` results terminate as
`suppressed`. Explicit mentions are accepted without invoking that classifier. The participant rule
applies to any eligible non-guest human participant, not only the original requester.
`explicit_mentions_only` disables candidate ingestion and requires a mention on every turn.

The classifier sees the candidate message as untrusted data and returns only an enum. It is not a
retrieval or authorization layer, and it is deliberately outside Slack's acknowledgement path.
An explicit mention inside an existing Slack thread creates a new run in that same conversation;
it does not create a nested thread or second Agent Session. See
[Slack thread and Agent Session model](thread-and-session-model.md) for the complete identity and
ordering contract.

## Durable execution

Inngest owns durable retries, child invocation, capacity limits, cancellation, and side-effect
ordering. PostgreSQL remains the causal and idempotency authority. `route-slack-turn`:

1. inserts or reuses one immutable `slack_turns` row for either incoming event type;
2. claims the earliest known pending row for that conversation;
3. invokes the responder classifier only for an ordinary follow-up candidate;
4. for an accepted turn, creates `agent_runs` and links it to the turn in one transaction;
5. invokes the progress child and then the primary question child with `step.invoke`;
6. verifies the linked run is terminal before completing the turn as `routed`.

`slack_turns` is the shared causal queue. Every insert, claim, link, and terminal transition takes a
transaction-scoped advisory lock keyed by `conversation_id`. Slack timestamps are parsed into
`NUMERIC(30, 6)` and ordered numerically, with creation time and event ID as deterministic tie
breakers. An immutable identity check rejects a replay that reuses an event ID with different
workspace, channel, user, message, thread, or turn-kind fields. A partial unique index allows only
one `processing` owner per conversation. Inngest's per-conversation constraint reduces contention,
but the database remains authoritative because durable sleeps and child invocations can release
Inngest capacity.

A later already-known turn retries the claim through at most 240 durable 15-second sleeps (about one
hour) and then fails into turn cleanup. The queue orders all rows known when the head is selected; it
does not preempt an already-`processing` row if a previously unseen event with an older Slack
timestamp arrives afterward. Preserving the existing processing lease prevents two owners from
executing concurrently, at the cost of strict timestamp order across that late-arrival boundary.

The normal accepted-turn path has no PostgreSQL-to-new-Inngest-event dual write: run creation/linking
is one database transaction inside the already durable router, and progress plus processing are
ordered child invocations. The primary question function then:

1. claims or resumes the ledger run;
2. reuses or claims progress as a backstop;
3. runs or resumes the graph, publishes monotonic sanitized stages, and persists the final result;
4. checks cancellation, installs an immutable hash manifest, and claims delivery;
5. publishes each final-answer part as its own ordered durable step;
6. completes delivery before marking the run successful.

The progress child has its own Slack capacity rather than consuming a model slot. Its non-idempotent
`chat.startStream` call remains split into database claim, remote start, and acknowledgement steps,
so a retry after the returned timestamp becomes durable does not repeat the start. Provider and Slack
transport retries are disabled so Inngest remains the durable retry owner.

Question, turn, and cancellation cleanup functions reconcile their respective state machines. Turn
cleanup cannot release a linked queue owner until its `agent_runs` row is terminal. If a cleanup
function itself exhausts retries, the final unreconciled-run function is the dead-letter boundary:
it terminalizes or abandons the linked run first and then completes the turn, preventing one poisoned
`processing` row from blocking the conversation forever. Router and cleanup owners subscribe to both
Inngest failure and cancellation system events, including timeouts. A cleanup waiting behind an
earlier turn rechecks with fresh durable claim steps rather than replaying one stale blocked result.
The exact eight-function inventory is in
the [implementation journal](implementation-journal.md#current-function-inventory).

That split does not make progress creation exactly once. If the process exits after Slack creates a
stream but before Inngest durably records the start-step result, neither the ledger nor Slack offers
a client-key lookup that can recover the timestamp. Inngest may retry the non-idempotent step and
create a second surface while the first remains orphaned. Once the timestamp is durable, later
PostgreSQL acknowledgement retries are safe; before it is durable, this is an unavoidable
cross-system at-least-once window. Final-answer idempotency is a separate ledger-backed guarantee.

## Slack progress, finalization, and Stop

The primary UX uses `chat.startStream` in plan mode and `chat.appendStream` task updates. Graph node
boundaries map to fixed, code-owned labels: understanding, searching, reviewing, drafting,
verifying, and tightening. No model scratchpad, retrieved text, or unverified answer tokens are sent
as progress. After grounding verification, the complete answer—and sources when requested—is
supplied to `chat.stopStream` as the canonical first part. Responses over the 12,000-character
stream limit are split at readable boundaries and posted as deterministic, ordered continuation
messages.

Artifact citations remain internal provenance and are persisted for evaluation and delivery safety.
Slack hides citation markers and the source list by default; the retrieval planner enables them only
when the user directly asks for sources, citations, evidence, provenance, or supporting documents.

Each newly reached stage becomes the one active task; the preceding task is completed under its
stable ID, so Slack can keep completed work visible while presenting the latest work first. The plan
title follows the current stage rather than repeating Slack's session-level "is working" loader.
On completion, the last active task is marked complete under its original label, a visible
`Answer ready` task is added, and the plan title becomes `Answered in …`, using the
stream's persisted start timestamp. Exact plan animation and completed-task ordering remain Slack
client behavior.

Native streaming and Agent Sessions are the supported Slack surface. The initial task is the neutral
`Understanding the request`; search/review labels appear only after the graph actually takes those
paths. If opening a native stream
fails or its response is ambiguous, the publisher never creates an editable progress placeholder.
It suppresses task updates, keeps the thread-scoped session state best-effort, and posts the verified
final answer once with a deterministic message ID. Progress is best-effort; final delivery remains
strict and ledger-backed. An ambiguous `chat.stopStream` is retried with the identical canonical
answer chunks and canonical delivery metadata. `message_not_in_streaming_state` is not accepted as
proof because a user Stop may have ended a progress-only stream. The publisher verifies the metadata
through bounded read-back. Metadata must remain absent from the exact message across every poll
before the publisher uses a deterministic immutable final post; a later match, failed read, or mixed
result is retried rather than guessed. It does not rewrite the finalized message through
`chat.update`.

Agent Sessions supply Slack's native processing loader and Stop control. The processing row is
Slack client chrome: the default copy is `{bot display name} is working…`, and native Stop is
revealed on hover of that row. `agents.sessions.setStatus` has no display, hover, or loading-copy
argument, so the app leaves that loader as Slack draws it rather than emulating Stop with a second
control. A verified
`agent_session_stopped` event atomically records an event-to-run claim before sending the Inngest
cancellation event. Final delivery claim and Stop lock the same run as their linearization point:
cancellation prevents delivery if it commits first, while a `delivering` run rejects a late Stop and
continues to the verified answer. Cancellation is cooperative: Inngest stops between steps and the
graph/publisher check the ledger flag between observable stages and before final delivery. An
in-flight provider request may finish, but a cancellation that wins before delivery produces a
stopped notice rather than an answer. The primary function can finalize an observed Stop itself, so
loss of the cancellation-event handoff cannot leave a run processing forever. System, timeout, and
administrative Inngest cancellations without a persisted Slack Stop produce a neutral failure rather
than falsely attributing the action to the user. Terminal session activation is a required
retryable effect when final delivery did not itself close the stream.

Cleanup does not use session activation as a generic reset. A stale or rejected Stop, an event with
no matching run, and cleanup that observes an already-succeeded run leave the session unchanged.
The successful finalization that delivered the outcome is the owner of `active` versus `suspended`;
in particular, delayed Stop cleanup cannot overwrite a clarification's `suspended` status.

The timestamp returned by `chat.startStream` is acknowledged under the run lock even if Stop wins
during that network call. When the result reaches Inngest, this preserves enough identity to close
the already-created remote stream without publishing another Stop confirmation. Cancellation
cleanup alone owns the user-visible terminal notice. This cannot cover a process exit before
Inngest persists the result, because that exit loses the only returned stream identity.

If a verified result was persisted before the primary function exhausted retries, failure cleanup
reconstructs the same immutable manifest and resumes every unacknowledged part before considering a
generic error. If all parts are already acknowledged, it promotes the run to delivered without
reposting. If only some long-answer parts were
acknowledged, it preserves those verified parts and appends a deterministic incomplete-delivery
notice rather than overwriting them with an unrelated error.

## Bounded answer workflow

The LangGraph workflow performs:

1. deterministic whole-message greeting recognition;
2. follow-up resolution using bounded conversation history;
3. structured intake and retrieval planning in one model action;
4. direct finalization for greetings, clarification requests, and out-of-scope requests;
5. structured account lookup and/or lexical search for a knowledge question;
6. artifact reads within count and character budgets;
7. evidence grading;
8. at most one retrieval refinement under the production profile;
9. answer generation;
10. citation and grounding verification;
11. at most one repair followed by re-verification;
12. finalization or a clear insufficient-evidence response.

Evidence is preserved across refinement rounds, deduplicated by artifact ID, and kept within the
context budget. The graph also has an explicit recursion limit, so no path can become an open-ended
agent loop. Clarification responses carry a typed disposition through persistence and leave the
Slack Agent Session `suspended`; other completed outcomes leave it `active`.

### Semantic planning, deterministic execution

The model decides semantic intent through one structured `RetrievalPlan`. For a knowledge question
it returns a typed `response_mode`, retrieval queries or account filters, and optionally a
`reuse_turn_id` selected from the bounded prior turns supplied by the application. This is semantic
selection, not authorization: code rejects an unknown turn ID, prevents a source-only response from
triggering retrieval, and enforces every action budget.

| response mode | intended use | deterministic execution |
| --- | --- | --- |
| `answer` | A new or contextual knowledge answer | Re-read selected prior artifact IDs, then execute only the planned gap retrieval |
| `sources_only` | Provenance for an earlier answer | Render the selected turn's saved source references with no retrieval or answer-generation call |

This avoids phrase-specific routing such as checking for a literal "show sources" string. New
wordings remain a model classification problem, while state access and side effects remain typed,
bounded code. Adding a genuinely new response behavior requires one reviewed schema value and one
deterministic execution path rather than another collection of regex branches.

### Conversation state and provenance ownership

`conversation_id` is the LangGraph `thread_id`, so one Slack root thread owns one checkpointed
conversation and separate threads remain isolated. Current-run fields such as `evidence`, queries,
drafts, and counters are reset for every accepted question. The bounded `history` list survives and
stores one compact record per turn:

- stable run ID and original question;
- clean displayed answer, with internal artifact citation markers removed;
- compact source references for artifacts actually cited by that answer;
- ordered IDs for the artifacts retrieved by the turn.

History does not duplicate full artifact content or snippets. With the production limits it retains
at most six turns and at most sixteen artifact IDs per turn. A source request can therefore render
saved provenance without I/O. A substantive contextual follow-up deterministically re-reads known
IDs from immutable SQLite, merges them with current evidence, and searches only for material gaps.
Re-reading preserves current database truth and avoids treating checkpointed model text as evidence.

### Action budgets

The production profile permits at most eight retrieval tool calls per question. The longest
ordinary path is bounded as follows:

| retrieval round | maximum calls |
| --- | ---: |
| Initial: account lookup, three lexical searches, one artifact batch read | 5 |
| Refinement: two lexical searches, one artifact batch read | 3 |
| Total | 8 |

A prior-evidence read counts against the same ceiling. The executor reduces new search fan-out when
reuse consumes a call; it never hides a ninth tool action inside an existing call. The model-call
ceiling is nine, covering the longest legal history-resolution path, one structured-plan repair,
two evidence grades, answer generation, grounding verification, one answer repair, and
re-verification. These ceilings bound failure and cost; retrieval shape and model selection are
evaluated separately rather than silently increasing the limits.

### Failure behavior

Malformed structured output is re-requested once within the model budget. A second invalid result,
an unavailable prior-turn ID, or an exhausted model budget raises a typed terminal failure. Inngest
marks deterministic validation failures non-retriable, records a stable error code and correlation
identifiers, and lets cleanup publish the code-owned safe Slack message. Logs do not include raw
provider errors, prompts, evidence, secrets, or user-visible stack traces.

### Retrieval coverage and scaling properties

FTS5 still computes BM25 globally. When diversification is enabled, the repository over-fetches a
bounded candidate set, takes a profile-controlled number per scenario on the first pass, and then
backfills in original BM25 order. Backfill preserves recall for narrow single-scenario questions;
the first pass prevents a heavily documented scenario from immediately occupying every result.
Global BM25 and first-pass values of one, two, and three are separate fixed-model evaluation
profiles, so the production value can be selected from retrieval recall and answer quality rather
than intuition.

Runtime growth is bounded by profile values, not by thread length or corpus density: history,
queries, candidates, artifact reads, context characters, retrieval rounds, tool calls, model calls,
and graph recursion all have explicit ceilings. Scaling beyond the current single-corpus design
would require authorization filters and an indexed corpus strategy, but it does not require a new
follow-up graph for every wording or source request.

## Data ownership

- The supplied SQLite database is immutable knowledge input. It is opened with SQLite URI
  `mode=ro`, `immutable=1`, and `PRAGMA query_only=ON`.
- Alembic owns the application tables for causal turns, runs, streams, delivery, cancellation, and
  cited-source state. `slack_turns` is one immutable event/queue row; `agent_runs` is the linked run
  aggregate. Repeated facts are atomic child rows (`run_sources`, `run_delivery_parts`, and
  `slack_stopped_streams`); the typed immutable result snapshot is the only JSON replay value and is
  not a relational query surface.
- LangGraph's supported Postgres checkpointer owns its own checkpoint tables.
- Inngest owns durable function state and retry scheduling.
- Local Langfuse tracing is optional observability, not runtime or evaluation source of truth.

The current single-workspace product treats the supplied SQLite database as one corpus visible to
eligible non-guest workspace members in channels containing the bot. Slack does not permit workspace
guests to access apps with the Agents feature enabled, so guest access is outside this assumption.
There is no row-, customer-, or role-level authorization model in the provided data. If the corpus
becomes permissioned, access constraints must enter every retrieval path and become part of
delivery, evaluation, and any future cache key.

The local runtime persists Slack-derived state in several places: identifiers and timestamps in the
application ledger, questions and answers in LangGraph checkpoints, final answer snapshots, and
Inngest development event/run state. Optional remote tracing may also receive graph or model inputs
when explicitly enabled. The current repository has no automated TTL, Slack-message-deletion
synchronization, or user/workspace erasure workflow. Limiting the number of turns sent back to the
model bounds context size but does not delete persisted state.

That retention behavior is a local take-home limitation. Slack's
[Agent guidance](https://docs.slack.dev/ai/developing-agents/#data-retention) recommends storing
metadata and fetching Slack content in real time rather than storing Slack data. A production
deployment must first classify every Slack-derived field, minimize persisted content, define and
enforce retention windows, propagate message deletion and installation revocation, and verify
erasure across the ledger, checkpoints, Inngest state, traces, caches, backups, and disaster-recovery
copies. Local Compose volume deletion is not a substitute for that lifecycle.

The application tables contain internal operational state and are reachable only through the
trusted service repository; there is no end-user SQL/API access path and no tenant column on which
an honest row-level-security policy could operate. PostgreSQL RLS is therefore not enabled merely
for appearance. A deployed database should use separate least-privilege migrator and runtime roles.
If tenant- or customer-level authorization is introduced, the schema must first add an explicit
authorization dimension and enforce it in retrieval and runtime tables before adding and forcing
tested RLS policies.

## Health and observability

- `GET /healthz` reports process liveness without checking dependencies or exposing secrets.
- `GET /readyz` directly checks the knowledge file and PostgreSQL connectivity after strict
  configuration validation. It does not report Slack, OpenAI, or Inngest as ready merely because
  settings are present.
- Local Compose runs Inngest's bundled `inngest alpha doctor healthcheck`, and the Slack-only ingress
  waits for both application liveness and Inngest health before starting. This is a startup gate,
  not a continuous external-service availability guarantee.
- Structured logs carry request, run, and conversation identifiers.
- PostgreSQL retains run status, model/tool/retrieval action counts, latency, model/version metadata,
  and sources.
- The Inngest UI exposes durable steps and retries.
- Langfuse tracing can be enabled explicitly for local model and graph traces.
