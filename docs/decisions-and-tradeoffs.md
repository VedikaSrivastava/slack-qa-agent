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

## Inngest for durable background work

**Decision:** acknowledge Slack after the short durable enqueue step, then move model work and Slack
updates into a bounded Inngest function.

**Why:** model calls can outlive Slack's acknowledgement window. Inngest provides retries,
idempotency, per-conversation serialization, and observable steps without coupling those concerns
to LangGraph.

**Tradeoff:** there is one more local service and a hosted dependency in deployment. The graph is
kept as one coarse durable step so Inngest does not duplicate graph-level orchestration.

## PostgreSQL for application state and checkpoints

**Decision:** use PostgreSQL for the run ledger and LangGraph conversation checkpoints.

**Why:** durable Slack delivery, idempotency, concurrent workers, and multi-turn state need shared
transactional storage. SQLite remains dedicated to the supplied immutable knowledge dataset.

**Tradeoff:** a database service is required locally. A hosted database such as Supabase could
provide PostgreSQL in production without changing the storage interfaces; Turso would be a more
substantial change because the current persistence and checkpoint integrations are PostgreSQL
specific.

## Alembic migrations are Python files

**Decision:** manage application-owned SQLAlchemy tables with Alembic revisions.

**Why:** Alembic revisions are executable Python because they use its migration API, track ordered
upgrade/downgrade history, and remain aligned with SQLAlchemy metadata. Offline mode can still render
the complete migration chain as SQL for review.

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

## Code-defined versions and optional LangSmith tracing

**Decision:** keep model names, prompt/retrieval versions, and agent budgets in reviewed code.
Require LangSmith only for hosted dataset and experiment commands; keep runtime tracing off by
default.

**Why:** a checkout is self-contained without access to a private LangSmith prompt or project.
Traces are useful evidence, but the application must not depend on them to answer Slack questions.

**Tradeoff:** changing a model or prompt requires a code review and deployment. Evaluators without
LangSmith access can still run the local suite, but they cannot see hosted experiment history unless
it is shared separately.

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
