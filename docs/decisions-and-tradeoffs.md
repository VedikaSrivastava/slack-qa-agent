# Engineering decisions and tradeoffs

These notes record the reasoning behind the checked-in implementation. They describe current
choices rather than requirements for every future version.

## Slack is the only product interface

**Decision:** expose Slack Events API ingress and keep the CLI as a developer diagnostic only.

**Why:** the assignment is Slack-based, and a local web UI would create a second product surface to
build, test, and maintain without exercising Slack authentication, event delivery, threads, or
retries.

**Tradeoff:** end-to-end local testing needs a Slack workspace and an HTTPS tunnel. The Slack web or
desktop client supplies the UI; no Slack server runs locally.

## HTTP Events API instead of Socket Mode

**Decision:** receive signed events at `/slack/events`.

**Why:** Slack's
[HTTP and Socket Mode comparison](https://docs.slack.dev/apis/events-api/comparing-http-socket-mode/)
recommends Socket Mode for convenient local development and HTTP request URLs for production
reliability. HTTP is stateless, scales horizontally behind standard infrastructure, maps directly
to FastAPI, and uses the same ingress path locally and in deployment. Slack Bolt handles URL
verification and request signing.

**Tradeoff:** local testing needs a public HTTPS tunnel. The tunnel targets a small exact-path proxy
that forwards only `POST /slack/events`; it does not publish health, readiness, or the unsigned local
Inngest development endpoint. Socket Mode is a valid alternative and could be implemented with
Bolt's Socket Mode adapter plus an app-level token. Supporting both transports would remove the
tunnel locally but add another credential, a stateful WebSocket lifecycle, and a second ingress path
to configure and test. This repository intentionally keeps one production-like HTTP adapter rather
than maintaining environment-specific Slack transports.

## Native task streaming, but verified final answers

**Decision:** use Slack's `chat.startStream`, task chunks through `chat.appendStream`, and
`chat.stopStream` as the primary response surface. Stream fixed observable work stages, then publish
the complete grounded answer only after verification.

**Why:** users get genuine progress—understanding, searching, evidence review, drafting, and
verification—without exposing private chain-of-thought, retrieved content, or a plausible-looking
draft that may later fail grounding. Slack receives coherent answer blocks rather than a cosmetic
character-by-character animation. Agent Sessions add the native processing state and Stop control.

**Tradeoff:** the final prose appears as one verified chunk rather than token by token. A response
above Slack's stream-markdown limit becomes one finalized primary part plus ordered continuation
messages. If native stream opening fails or is ambiguous, progress is suppressed and the verified
answer is posted once with a deterministic message ID. The runtime does not emulate streaming with
an editable placeholder. The stream starts with `Understanding the request`; research labels are
emitted only when the graph actually searches or verifies.

## Cooperative Stop instead of unsafe interruption

**Decision:** atomically claim each Slack Stop event and bind it to its original run, persist
cancellation intent, emit a matching Inngest cancellation event, check the flag between graph stages
and before delivery, and use both primary-path and dedicated cleanup finalizers for the Slack
surface.

**Why:** Slack events, Inngest cancellation, model calls, and message writes have different failure
boundaries. A persisted event-to-run result makes Slack retries harmless. Stop and final-delivery
claims lock the same run: whichever commits first wins, so delivery never starts after an accepted
Stop and a late Stop never interrupts an answer already being finalized. Cleanup makes the selected
nonterminal run's stream/session terminal even when the primary function is cancelled between
steps; the primary path performs the same idempotent finalization when it observes the flag directly.

**Tradeoff:** cancellation cannot abort an already executing provider HTTP request mid-flight; that
call may finish before the graph observes cancellation. The result is withheld when cancellation
wins before delivery, and the user sees a concise stopped notice. Timeouts or administrative Inngest
cancellation without an accepted Slack Stop use a neutral failure message instead.

Slack may stop a stream before its Stop event reaches the application. Therefore an
`already stopped` API error is not treated as answer-delivery proof. Final chunks include canonical
run/hash metadata; the publisher verifies that identity through bounded read-back and posts a
deterministic immutable answer or stopped confirmation only after seeing the exact message without
the metadata on every bounded poll. A later match, failed read, or mixed observation remains unknown
and retries. This favors a visible canonical outcome over falsely acknowledging a progress-only
message or duplicating a write whose response was lost. A late stream-start acknowledgement closes
that progress surface without content; cancellation cleanup is the only path that owns the visible
Stop confirmation.

Cleanup is not allowed to reset an arbitrary thread session to `active`. A rejected or stale Stop,
an event with no matched run, and cleanup that observes an already-succeeded run leave status
untouched. Successful finalization owns the outcome-specific status, including `suspended` for a
clarification, so delayed cleanup cannot overwrite it.

## Conservative unmentioned follow-ups

**Decision:** an explicit mention establishes an agent-owned thread. Ordinary human replies in that
thread become durable candidates, and a structured responder judge authorizes only clear
agent-directed requests. Any eligible non-guest participant may ask; ambiguous messages stay
silent. A configuration switch can require explicit mentions on every turn.

**Why:** this supports natural multi-turn Q&A without letting the bot interrupt unrelated discussion.
Putting the candidate behind the shared turn router keeps ownership lookup and model latency outside
Slack's acknowledgement window. The router invokes classification as a separate function under the
same global model-capacity constraint used by question processing.

**Tradeoff:** the conservative judge can miss a terse ambiguous follow-up. That is preferable to
responding into human conversation. The classification adds one bounded model call only for an
ordinary message in an already agent-owned thread; explicit mentions bypass it.

## Thread-scoped conversations and sessions

**Decision:** use workspace, channel, and root thread timestamp as the conversation identity for
both LangGraph checkpoints and Slack Agent Sessions. A mention inside that thread creates a new run
in the same conversation; it cannot create a nested Slack thread.

**Why:** Slack replies always point to the root `thread_ts`, and current Agent Sessions are
thread-scoped. Using the same stable identity keeps retrieval context, responder ownership,
progress ordering, and Stop handling aligned.

**Tradeoff:** all channel participants may see and continue the shared session under the current
single-corpus authorization assumption. A requester-private conversation would need DM support or
a separate product policy, not a different interpretation of Slack thread timestamps.

## Intake classification at the existing planning boundary

**Decision:** short-circuit obvious greetings deterministically, and extend the structured
retrieval plan with knowledge, clarification, greeting, and out-of-scope dispositions.

**Why:** `hey` should not search the database, an unclear request should not guess an entity, and an
unrelated request should not produce a misleading insufficient-evidence answer. Reusing the planner
keeps normal knowledge questions at the same model-call count.

**Tradeoff:** semantic classification can still make mistakes. Direct conversational replies are
code-owned, do not make knowledge claims, and bypass retrieval/citation checks; only knowledge
questions enter the grounded answer path. Clarification leaves the Agent Session `suspended` until
the next user turn.

## Inngest for durable background work

**Decision:** acknowledge Slack after the short durable event handoff, then route mentions and
ordinary follow-up candidates through one bounded `route-slack-turn` function. Accepted work invokes
progress and primary processing as ordered child functions.

**Why:** model calls can outlive Slack's acknowledgement window. Inngest provides retries, durable
sleeps and child invocations, capacity controls, timeouts, cancellation, and observable steps without
coupling those concerns to LangGraph.

**Tradeoff:** there is one more local service and a hosted dependency in deployment. The graph stays
one coarse durable step, while eight Inngest functions exist only at distinct routing, progress,
processing, and cleanup boundaries.

Inngest is intentionally not used as a replacement for PostgreSQL or as a wrapper around every read.
It owns durable invocation, retries, global model capacity, ordered Slack delivery steps, and
failure/cancellation triggers. LangGraph owns bounded reasoning and checkpoints. PostgreSQL owns the
`slack_turns` causal queue plus run, stream, delivery, cancellation, and Stop-event idempotency truth.

Failure cleanup prefers the durable verified result over a generic error: it reconstructs the
immutable delivery manifest and retries unacknowledged canonical answer parts. A safe error is used
only when no persisted result can be recovered. Inconclusive Slack read-back remains retryable;
there is no safe way to infer whether an ambiguous remote write committed while Slack history is
unavailable.

Both input event types first become immutable `slack_turns` rows. Transaction-scoped advisory locks,
numeric Slack timestamp order, and a partial unique index establish one `processing` owner per
conversation. Explicit mentions bypass the separately model-bounded classifier; ordinary candidates
invoke it only after reaching the head. Only accepted turns create an `agent_runs` row, and creation
plus the immutable turn link share one transaction.

The router invokes progress and then primary processing with `step.invoke`; it does not commit a run
and separately send a new question event. It completes a linked turn only after the run is terminal.
Later known turns use at most 240 durable 15-second sleeps (about one hour) before cleanup. One
intentional limit remains: an already-claimed turn is not preempted when a previously unseen, older
Slack event arrives. Inngest's per-conversation constraint reduces contention, but PostgreSQL is the
correctness boundary because capacity can be released during sleeps and child invocations.

Dedicated question, turn, and cancellation cleanup functions reconcile failures. If they exhaust
retries, a final dead-letter function terminalizes or abandons the linked run before releasing the
turn, so a poisoned queue owner cannot block the conversation indefinitely.

Stream creation itself is split into database claim, remote Slack start, and database
acknowledgement steps. Once Slack returns a timestamp, Inngest persists it as step output before the
ledger write, so retrying a failed acknowledgement cannot repeat the non-idempotent start call. An
ambiguous start response still degrades because Slack provides no application idempotency key for
that method.

This is not an exactly-once claim for stream creation. There is an unavoidable crash window after
Slack creates the stream but before Inngest persists the start-step output. If the worker exits in
that interval, a retry has no idempotency key or lookup by client ID with which to recover the first
timestamp, so it may create a duplicate while the first surface remains orphaned. The split removes
the later database-acknowledgement retry hazard; it cannot make a non-transactional Slack side effect
atomic with Inngest persistence.

## PostgreSQL for application state and checkpoints

**Decision:** use PostgreSQL for the run ledger and LangGraph conversation checkpoints.

**Why:** durable Slack delivery, idempotency, concurrent workers, and multi-turn state need shared
transactional storage. SQLite remains dedicated to the supplied immutable knowledge dataset.

**Tradeoff:** a database service is required locally. A hosted database such as Supabase could
provide PostgreSQL in production without changing the storage interfaces; Turso would be a more
substantial change because the current persistence and checkpoint integrations are PostgreSQL
specific.

The application-owned tables are internal service state, not tenant-queryable product tables. The
current corpus and run model have no tenant or permission dimension, so enabling row-level security
now would either block the service or create a policy the service role bypasses. Production should
separate migration and runtime roles and grant only required table/sequence operations. Real RLS
belongs with a future explicit tenant/authorization key, forced policies, retrieval enforcement,
and cross-tenant tests.

The state-machine fields that must linearize together remain one flat run aggregate under one row
lock. `slack_turns` is a separate atomic row per incoming event because queue lifecycle and optional
run linkage are independent of answer delivery. Repeated sources, delivery parts, and Stop-reported
stream identities are normalized into one atomic row per fact. The only JSON field is a bounded typed
result snapshot used as an immutable retry value rather than as a queryable document store.

## Slack-derived state retention remains a production requirement

**Decision:** describe the current persistence lifecycle as local-development behavior, not as a
production-compliant retention or erasure design.

**Why:** retries and multi-turn resolution currently persist Slack identifiers, questions, answers,
and derived execution state across the application ledger, LangGraph checkpoints, and Inngest. That
is useful for a durable take-home demonstration, but Slack's
[Agent guidance](https://docs.slack.dev/ai/developing-agents/#data-retention) recommends retaining
metadata and retrieving Slack content in real time instead of storing Slack data. A bounded prompt
history is not equivalent to deleting durable copies.

**Tradeoff:** this repository does not add a speculative purge subsystem without a product retention
period, subject identity contract, or production stores. Before deployment, those requirements must
be defined and implemented across checkpoints, ledger rows, durable-work payloads, traces, caches,
and backups, including Slack message deletion, user/workspace erasure, and app-uninstallation paths.
The current local stack has no automated TTL or erasure workflow.

## Alembic migrations are Python files

**Decision:** manage application-owned SQLAlchemy tables with Alembic revisions.

**Why:** Alembic revisions are executable Python because they use its migration API and remain
aligned with SQLAlchemy metadata. Because this repository has not been deployed or submitted, the
development revisions were squashed into one clean baseline instead of preserving artificial local
upgrade history. Future schema changes after a shared release should be additive revisions. Offline
mode can render the baseline as SQL for review.

**Tradeoff:** revisions are less immediately readable to someone expecting standalone `.sql` files.
LangGraph checkpoint tables are deliberately excluded because the supported checkpointer owns their
schema and setup.

## Immutable SQLite plus lexical and structured retrieval

**Decision:** use the supplied FTS5 index for lexical search and a parameterized relational lookup
for bounded account questions.

**Why:** this uses the schema and indexes already present in the assignment data, stays deterministic,
and handles both direct document questions and exact multi-account filters without exposing arbitrary
SQL to the model.

**Tradeoff:** lexical retrieval is weaker for semantically distant paraphrases. Embeddings or a
vector database should be introduced only if measured evaluations show that the added indexing,
deployment, and consistency costs improve the target cases.

The original delivery archive and extracted database are retained under `data/` for simple setup
and clear provenance. SQLite `-wal` and `-shm` runtime sidecars are ignored rather than committed.
Docker copies only the main database into the validation stage so the supplied-schema integration
test runs, while the runtime image receives that file through a read-only bind mount.

## Bounded LangGraph instead of an open agent loop

**Decision:** code-review the model, retrieval limits, tool-call budget, retrieval rounds, and repair
count in named agent profiles.

**Why:** bounded paths make latency, cost, failure modes, and experiment configurations reviewable
and comparable. They also prevent the model from silently expanding work based on an environment
variable.

**Tradeoff:** unusually broad questions can exhaust the configured budgets and return insufficient
evidence. Profiles can be compared through experiments before any production budget is changed.

## Evidence-first generation and explicit grounding checks

**Decision:** never generate an answer without retrieved evidence; verify citations after generation
and after the single repair attempt.

**Why:** the bot should fail clearly instead of presenting unsupported knowledge-base claims.

**Tradeoff:** an answer may be rejected even when it sounds plausible, and verification adds model
latency. This is intentional for a grounded Q&A system.

## Docker Compose as the recommended local launcher

**Decision:** use Compose to pin and start the app, path-restricted Slack ingress, PostgreSQL,
migrations, and Inngest.

**Why:** evaluators get one consistent command without installing matching versions of Python,
PostgreSQL, or the Inngest CLI.

**Tradeoff:** Docker Desktop consumes more resources than native processes. It is a local packaging
choice, not an architectural requirement; deployed services can use the same container with managed
PostgreSQL and hosted Inngest. The HTTPS tunnel remains a separate host process.

The Slack-only ingress is startup-gated on both the application health check and Inngest's bundled
`inngest alpha doctor healthcheck`. This verifies that the local durable-work service is responding;
an `INNGEST_BASE_URL` value alone is only configuration. The application `/readyz` endpoint reports
only the knowledge-file and PostgreSQL probes it actually performs, and the Compose dependency is a
startup ordering control rather than continuous failover.

## Code-defined versions and optional LangSmith tracing

**Decision:** keep model names, prompt/retrieval versions, and agent budgets in reviewed code.
Require LangSmith only for hosted dataset and experiment commands; keep runtime tracing off by
default.

**Why:** a checkout is self-contained without access to a private LangSmith prompt or project.
Traces are useful evidence, but the application must not depend on them to answer Slack questions.

**Tradeoff:** changing a model or prompt requires a code review and deployment. Evaluators without
LangSmith access can still run the local suite, but they cannot see hosted experiment history unless
it is shared separately.

## No answer cache without evidence that it helps

**Decision:** do not cache generated answers in the take-home runtime. Treat caching as a measured
production extension, not a correctness shortcut.

**Why:** repeated wording does not guarantee the same authorized evidence, knowledge snapshot,
conversation context, prompt/retrieval version, or expected answer. The supplied corpus is small,
and the current evaluation has not demonstrated that repeated-query load is a bottleneck worth the
consistency and security surface.

**Tradeoff:** an identical repeated question currently performs another bounded agent run. If
production measurements justify caching, the design must first define an authorization-aware key
(workspace/tenant and permission scope, normalized question, relevant conversation context,
knowledge snapshot, model, prompt, retrieval, and answer-format versions), TTL, invalidation on
corpus or permission change, encrypted storage, source revalidation, stampede control, and safe
failure behavior. Cache-hit rate, latency, token savings, staleness, and authorization-isolation
tests should determine whether it remains enabled. Results must never cross authorization scopes.

## Current limitations

- Retrieval is intentionally lexical and schema-aware rather than semantic/vector-based.
- The official benchmark contains only seven human-curated cases and should not be treated as broad
  production coverage.
- Model aliases identify reviewed model families but do not pin provider weights. Saved experiment
  metadata records the exact configured names, and model-dependent results should be rerun before a
  production-profile change.
- Unit tests do not replace a live Slack, Inngest, PostgreSQL, OpenAI, and tunnel integration test.
- A single-workspace bot token is sufficient for the take-home; public multi-workspace distribution
  would require a complete OAuth installation flow and token storage.
- The SQLite bundle is treated as one internal corpus available to eligible non-guest workspace
  members in channels containing the bot. Slack Agent apps are unavailable to workspace guests.
  Customer-, tenant-, and role-level retrieval authorization require a data model and policy
  contract that the supplied database does not provide.
- Slack-derived state has no automatic local TTL, deletion synchronization, or end-to-end erasure
  workflow. Production requires data minimization and a verified retention/erasure lifecycle across
  every durable and observability store.
- Causal ordering is exact among persisted turns, but an already-`processing` turn is not preempted
  by a previously unseen older Slack event. A known blocked turn waits for at most about one hour
  before the cleanup path takes over.
- Agent Sessions and native stream rendering require the Slack workspace to enable the Agent
  feature. A stream-open failure still permits one immutable final post, but there is deliberately
  no editable progress-placeholder mode. Live Slack behavior needs an integration test in the
  evaluator's workspace.
- `chat.startStream` has no application idempotency key. A process crash after Slack creates a
  stream but before Inngest stores the returned timestamp can leave an orphan progress surface and
  allow a retry to create another. The runtime guarantees neither exactly-once progress creation nor
  recovery of that unknown timestamp; final-answer delivery uses a separate persisted protocol.
