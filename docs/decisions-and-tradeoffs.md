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

### Classifier cost and noisy threads

The responder judge uses the production profile's router role, currently `gpt-5.4-nano` under
`split-gpt-5.4-hybrid`. It sends a fixed routing prompt, the new message (at most 8,000 characters),
an optional latest clarification question, and — under the `LATEST_AGENT_CONTEXT` variant that
production selects — the latest agent response, also bounded to 8,000 characters. It never sends
the whole Slack thread. A normal candidate makes one short structured-output call; Inngest may
retry a failed classifier function at most twice. Explicit mentions cost no classifier call.

The latest agent response was added because a terse `please try again` or `can you redo that` has no
referent on its own, so the classifier could not distinguish a retry request from unrelated human
conversation. It is labelled untrusted, is used only to judge whether a terse message continues the
agent's own turn, and explicitly does not override a clear human-to-human exchange, an
acknowledgement, a logistics note, or a request directed at another person. It roughly doubles
classifier input tokens on threads with long prior answers, which is the accepted cost of the
capability.

Routing is a three-way decision over a short input, so it is deliberately assigned the cheapest
model in the profile rather than the answer model. Exact per-decision cost depends on message
length, tokenization, retries, and the configured profile; the provider's model pricing page is the
source of truth.

Rambling is deliberately not treated as a transcript to keep sending to the model. The classifier
sees only the candidate message and can remain silent. The answer workflow retains at most six
finalized semantic turns, while explicit recovery includes no more than three suppressed human
messages and 4,000 characters. Older discussion falls out of model context rather than increasing
cost indefinitely. Users should restate the relevant question or start a new thread when context
has drifted; a future production policy could add an aggregate token cap or a user-confirmed
summary, but neither should silently replace the current bounded evidence-first behavior.

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

## Typed response modes with deterministic prior-turn reuse

**Decision:** let the structured planner semantically select `answer` or `sources_only` and an
optional prior-turn ID, then validate and execute that plan in deterministic code.

**Why:** follow-up language is open-ended, so literal source-request phrases and question-specific
branches do not scale. The model is useful for deciding which earlier topic the user means. It must
not be responsible for dereferencing arbitrary IDs, deciding budgets, or inventing unavailable
history. A typed plan keeps that security and reliability boundary explicit.

**Tradeoff:** planner quality can still select the wrong available turn. The bounded history summary
therefore exposes stable turn IDs, questions, and provenance availability, while code rejects IDs
that were not supplied. A new execution behavior must extend the reviewed schema; it cannot appear
silently from free-form model text.

## Compact per-turn provenance instead of checkpointed evidence copies

**Decision:** retain the clean answer, cited source references, and ordered retrieved artifact IDs
for each bounded conversation turn. Do not retain full evidence or snippets in turn history.

**Why:** source-only follow-ups need provenance, and contextual follow-ups need a reliable bridge
back to evidence. IDs provide that bridge without multiplying knowledge-base text across every
checkpoint. Deterministically re-reading those IDs from immutable SQLite restores trusted evidence
and permits gap retrieval.

**Tradeoff:** a later source-only request can show only sources actually saved for the selected
turn. If an artifact disappears in a future mutable corpus, contextual reuse must treat the missing
read as an evidence gap rather than trusting the historical answer.

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

Head is now `0002`, which adds `slack_turns.message_text` for Stop recovery. It was written as an
additive forward revision rather than folded into the baseline, both because the baseline was
already in the shared working tree and because it demonstrates the intended safe pattern for a
`NOT NULL` column on a populated table: add with a temporary server default, then drop the default
so later inserts must supply real text rather than silently accepting an empty string.

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

### Bounded scenario-first diversification

**Decision:** compare global BM25 with bounded candidate over-fetching and a configurable
per-scenario first pass, followed by BM25-order backfill.

**Why:** global BM25 can let one heavily documented scenario occupy most of top-K for a broad
cross-account question. A first pass improves coverage, while backfill prevents a hard per-scenario
cap from starving a legitimate narrow query.

**Tradeoff:** the first-pass number is not assumed to be universally correct. Global BM25 and
values one, two, and three are fixed-model retrieval profiles. Retrieval recall, citation recall,
answer correctness, action counts, latency, and cost must select the default; the official seven
cases remain immutable and robustness cases remain separate.

### Query-diverse comparison evidence

**Decision:** let the model plan several concise retrieval queries, but merge results round-robin
by planned query rather than globally sorting BM25 scores from unrelated queries.

**Why:** a comparative question often needs different evidence dimensions, such as competitor risk
and a promised milestone. SQLite scores are meaningful within a query, not across differently
worded queries. Preserving each query's strongest unique result makes the final evidence set more
representative without increasing the configured tool-call budget.

**Tradeoff:** this can retain a weaker result from one query instead of another result from a highly
productive query. Evidence grading and one bounded refinement remain responsible for identifying
gaps. A hybrid/vector retrieval change requires a measured robustness-suite recall improvement,
not this one failure alone.

The original delivery archive and extracted database are retained under `data/` for simple setup
and clear provenance. SQLite `-wal` and `-shm` runtime sidecars are ignored rather than committed.
Docker copies only the main database into the validation stage so the supplied-schema integration
test runs, while the runtime image receives that file through a read-only bind mount. The runtime
wheel excludes `knowledge_assistant.evals`, and the runtime stage does not copy tests, evaluation
datasets/reports, or the evaluation CLI.

## Bounded LangGraph instead of an open agent loop

**Decision:** code-review the model, retrieval limits, tool-call budget, retrieval rounds, and repair
count in named agent profiles.

**Why:** bounded paths make latency, cost, failure modes, and experiment configurations reviewable
and comparable. They also prevent the model from silently expanding work based on an environment
variable.

**Tradeoff:** unusually broad questions can exhaust the configured budgets and return insufficient
evidence. Profiles can be compared through experiments before any production budget is changed.

### One model per role rather than one model per profile

**Decision:** allow a profile to set a different model for resolve, plan, grade, verify, answer,
repair, and router. Production uses `split-gpt-5.4-hybrid`: `gpt-5.4` for plan, answer, and repair;
`gpt-5.4-mini` for grade and verify; `gpt-5.4-nano` for resolve and router.

**Why:** these nodes are not the same workload. Rewriting a pronoun and making a three-way routing
decision need instruction-following on a short input. Planning queries and synthesising a grounded
multi-hop answer decide whether the run succeeds. Grading and verification are schema-constrained
audits. Paying answer-model rates for a pronoun rewrite buys nothing, and paying nano rates for
answer synthesis loses the cases that matter.

**Tradeoff:** a profile now names several models, so a result is attributable to a combination
rather than a single model. Each unset override falls back to `model_name`, so single-model
profiles and their existing reports remain valid and comparable. Repair falls back to the answer
model rather than the default so a repair can never be written by a weaker model than the draft it
corrects. Temperature is resolved per model name because `gpt-5` models reject an explicit
temperature that `gpt-4.1` models accept. Trace metadata records both `model` (answer) and
`classify_model` so a saved experiment is not described by one misleading name.

The eight-call retrieval ceiling represents the longest configured two-round path: five initial
calls and three refinement calls. Reusing prior evidence consumes one of those calls and reduces new
search fan-out. The nine-call model ceiling covers the longest legal path including one structured
plan repair and one answer repair. These are honest safety ceilings, not targets the agent is
encouraged to spend.

## Evidence-first generation and explicit grounding checks

**Decision:** never generate an answer without retrieved evidence; verify citations after generation
and after the single repair attempt.

**Why:** the bot should fail clearly instead of presenting unsupported knowledge-base claims.

**Tradeoff:** an answer may be rejected even when it sounds plausible, and verification adds model
latency. This is intentional for a grounded Q&A system.

### Only grounding failures may reach the verifier

**Decision:** the grounding verifier holds unsupported claims, wrong exact values, dropped
qualifiers, and omitted requested facts. Answer wording and internal-scaffolding leakage are handled
by deterministic sanitization and generation instructions instead.

**Why:** verification escalates exactly once—`verify` to `repair` to `verify_repair` to
`reject_ungrounded_answer`—so a draft that fails twice becomes an abstention. A check placed there
can destroy a correct, well-grounded answer. Enforcing presentation through that gate was tried and
measurably converted a passing evaluation case into a flaky one that sometimes abstained outright.
Each concern now sits at the layer whose failure mode matches its severity: an exact textual
signature is stripped deterministically and cannot fail; a stylistic preference is a generation
instruction whose worst case is being ignored; only claim-level defects may abstain.

**Tradeoff:** a presentation rule the model ignores produces a slightly worse-worded answer with no
automatic correction, because nothing forces a retry. That is the intended trade: a correct answer
that reads imperfectly is strictly better than no answer. Deterministic stripping also cannot catch
the prose form of a leak—an answer that narrates retrieval mechanics in sentences rather than
brackets—so the generation instruction remains necessary and the two layers cover different halves
of the same defect.

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

## Code-defined versions and optional local Langfuse tracing

**Decision:** keep model names, prompt/retrieval versions, and agent budgets in reviewed code.
Keep runtime tracing optional and local evaluation independent of the tracing backend.

**Why:** a checkout is self-contained without access to a private hosted prompt or trace project.
Traces are useful evidence, but the application must not depend on them to answer Slack questions.

**Tradeoff:** changing a model or prompt requires code review and deployment. Local JSON reports are
portable, while optional Langfuse traces remain in the developer's configured environment.

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
- The production FTS first-pass value is still a code-reviewed candidate, not a measured winner;
  fresh retrieval and robustness matrices remain pending live API/data-transfer authorization.
- The model semantically selects among supplied prior turns and can choose the wrong available turn.
  Code prevents fabricated IDs and unsafe execution, but semantic selection quality still requires
  multi-turn evaluation.
- Provenance history is intentionally bounded. A source request for an evicted old turn cannot
  reconstruct sources unless another durable product record explicitly owns them.
- The official benchmark contains only seven human-curated cases and should not be treated as broad
  production coverage. Three repeats of those seven detect gross instability; they do not establish
  a production accuracy rate.
- `official-blueharbor-defection-risk` fails every repeat on the production profile, through
  `exact_dates` when it answers and `answerability_behavior` when it abstains. Its lexical coverage
  is far below every other case, which points at retrieval rather than presentation. No prompt
  revision in the v20-v22 sequence targeted it.
- Presentation rules in the generation prompt have no enforcement. A model that ignores them
  produces a worse-worded but still correct and grounded answer; only deterministic sanitization is
  guaranteed.
- `slack_turns` now stores human message text, not only identifiers. This is required for Stop
  recovery but increases the Slack-derived data footprint that the retention work below must cover.
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
