# Slack Q&A Agent

A production-minded Slack Q&A agent grounded in an immutable SQLite knowledge base. It uses
FastAPI and Slack Bolt for verified ingress, Inngest for durable background work, LangGraph for a
bounded retrieval workflow, PostgreSQL for application run state and conversation checkpoints,
and optional LangSmith tracing plus versioned offline experiments.

## Architecture

```text
Slack Events API
      |
      v
FastAPI + Slack Bolt -- deduplicate/create run --> PostgreSQL
      |
      +-- deterministic event ID --> Inngest
                                      |
                                      +-- mark running
                                      +-- create/reuse Slack placeholder
                                      +-- run QuestionProcessor (one durable step)
                                      |       |
                                      |       v
                                      |   bounded LangGraph
                                      |       |
                                      |       +-- typed search/read tools
                                      |       +-- read-only SQLite
                                      |       +-- Postgres checkpoints
                                      +-- update the placeholder
                                      +-- persist result and sources
```

The `QuestionProcessor` is transport-independent. Slack/Inngest, the CLI, and the evaluation runner
all invoke the same implementation. LangGraph owns question resolution, retrieval planning,
evidence grading, one optional refinement, generation, grounding verification, and one repair.
Inngest owns queueing, retries, per-thread ordering, a shared concurrency cap, and Slack side-effect
sequencing.

## Prerequisites

The recommended local path needs only Git and Docker Desktop. Python, uv, PostgreSQL, SQLite CLI,
Make, and the Inngest CLI are not required. Docker is a reproducible launcher, not an architectural
dependency: it runs the app, local PostgreSQL, and the pinned Inngest dev server. The HTTPS tunnel
used by Slack is a separate process. A deployed environment can use managed PostgreSQL and hosted
Inngest without Docker.

The supplied knowledge database is intentionally not committed. Place it at:

```text
data/synthetic_startup.sqlite
```

The application opens this file with SQLite `mode=ro`, `immutable=1`, and `PRAGMA query_only=ON`.
Docker mounts it read-only.

## Start with Docker

```bash
cp .env.example .env.local
docker compose --env-file .env.local up --build
```

On Windows PowerShell, use `Copy-Item .env.example .env.local` for the first command. The file asks
for the OpenAI key and two Slack credentials. LangSmith tracing is off by default; add a LangSmith
key and set `LANGSMITH_TRACING=true` when you want runtime traces. Local database, path, project,
and Inngest settings are fixed in Compose or code. Each process validates only the credentials it
uses and fails immediately when one is missing or blank.
`.env.local` is ignored by Git and excluded from Docker build contexts. Model names and action
budgets are code-defined in reviewed `AgentProfile` values, so environment variables cannot silently
switch either one. The production profile is currently the provisional `balanced-gpt-4.1-mini`;
run the saved experiments before treating that choice as final.

Services:

- Liveness: <http://localhost:8000/healthz>
- Readiness: <http://localhost:8000/readyz>
- Slack Events API: `http://localhost:8000/slack/events` (expose this with an HTTPS tunnel)
- Inngest UI: <http://localhost:8288>
- PostgreSQL: `localhost:5432` for local inspection only

`/healthz` reports process liveness without secrets. `/readyz` verifies the knowledge file,
PostgreSQL connectivity, and the validated Slack/agent/Inngest configuration. Slack is the only
product interface. Inngest always registers and executes the Slack question workflow.

The one-shot `migrate` service runs both `alembic upgrade head` and LangGraph's supported Postgres
checkpoint setup. Alembic owns only `agent_runs` and `run_sources`; LangGraph owns its
checkpoint tables.

## Developer-only direct agent check

```bash
docker compose --env-file .env.local run --rm app \
  python -m knowledge_assistant.cli ask \
  "For Verdant Bay, what is the approved live patch window?"
```

This command is a diagnostic for retrieval and agent development, not a product fallback and not a
substitute for the Slack integration. It requires only OpenAI and PostgreSQL, not Slack or LangSmith
credentials. To reuse multi-turn state across CLI calls, pass the same `--conversation-id` value.
Slack derives this value from team ID, channel ID, and the root thread timestamp, so different
threads are isolated and follow-ups in one thread share state.

## Evaluations

The committed `full` suite is the seven-question, human-curated assignment benchmark. It includes
reference answers, exact facts/entities/dates/commands/customer sets, verified source IDs, and
action budgets. A stable digest and the `official-v1` LangSmith dataset tag tie every experiment to
the exact gold data used. Synthetic questions are never written into this dataset.

The local `run` command requires OpenAI but does not require LangSmith. The `sync`, `experiment`,
and `augment` commands require `LANGSMITH_API_KEY` and fail when it is absent. Experiment execution
also fails if any evaluated run errors instead of silently producing a successful command exit.

Run the one-case smoke suite and save its local result:

```bash
docker compose --env-file .env.local run --rm app \
  python -m knowledge_assistant.evals run \
  --suite smoke \
  --profile balanced-gpt-4.1-mini \
  --output /app/evals/results/smoke-balanced-gpt-4.1-mini.json
```

Synchronize the versioned official dataset, then run and save a LangSmith experiment:

```bash
docker compose --env-file .env.local run --rm app python -m knowledge_assistant.evals sync
docker compose --env-file .env.local run --rm app \
  python -m knowledge_assistant.evals experiment \
  --profile balanced-gpt-4.1-mini \
  --protocol screening \
  --output /app/evals/results/langsmith-balanced-gpt-4.1-mini-screening.json
```

Repeat the experiment with `balanced-gpt-5-mini`, `balanced-gpt-5.6-luna`,
`lean-gpt-4.1-mini`, and `wide-gpt-4.1-mini`. These isolate model choice from retrieval/action
budget changes. Every run records deterministic pass/fail, semantic reference correctness from the
code-defined `gpt-5.6-terra` evaluator, source recall, action-budget compliance, latency, tokens,
estimated cost, errors, the complete profile, and trace links. Unknown profile names fail; there is
no model, evaluator, or budget fallback.

Use `screening` for one repetition across every profile. Run `confirmation` only for the top two;
it performs three repetitions per example to expose variance. Both protocols keep concurrency at
one so rate limits and latency comparisons remain controlled, and the protocol is saved in the
experiment metadata and local summary.

After the official baseline is stable, use a separate LangSmith augmentation-candidate dataset for
paraphrases and multi-turn variants. Curate insufficient-evidence cases independently rather than
deriving them from answerable gold seeds. Review and label all examples before promoting them into a
distinct robustness suite. Keeping generated examples separate avoids
optimizing against synthetic labels or contaminating the seven-question benchmark.

Generate two bounded candidates per official seed (14 total) with the code-defined generator:

```bash
docker compose --env-file .env.local run --rm app \
  python -m knowledge_assistant.evals augment --per-case 2
```

This command writes only to `slack-qa-agent-augmentation-candidates`, tags every example as
`review_status=candidate`, traces the generation, and raises if the model returns the wrong count.
It never edits or evaluates against `slack-qa-agent-official`; promotion requires human review.

## Slack setup

Create a Slack app with a bot token and signing secret, subscribe the Events API to
`https://<public-host>/slack/events`, and subscribe to `app_mention`. Grant the bot the minimum scopes
needed to read mentions and post/update thread messages (typically `app_mentions:read` and
`chat:write`). Set `SLACK_BOT_TOKEN` and `SLACK_SIGNING_SECRET`, then restart the app.

For local testing, run the Docker stack and expose port 8000 with an HTTPS tunnel such as ngrok or
Cloudflare Tunnel. Put the resulting HTTPS `/slack/events` URL in the Slack app's Event Subscriptions
page. The desktop Slack client is optional; Slack itself remains hosted and is not run in Docker.

Slack Bolt validates request signatures and timestamp freshness. Bot messages and irrelevant
subtypes are ignored. The HTTP handler creates an idempotent run keyed by Slack `event_id`, sends a
deterministically identified Inngest event, and returns without invoking a model. Retries reuse the
same run and Slack placeholder.

## Configuration

`.env.local` contains only external credentials and the explicit tracing switch:

- `OPENAI_API_KEY`: model access
- `SLACK_BOT_TOKEN`, `SLACK_SIGNING_SECRET`: Slack transport credentials
- `LANGSMITH_TRACING`: `false` by default; set to `true` only when runtime tracing is wanted
- `LANGSMITH_API_KEY`: optional for the Slack runtime and required for LangSmith commands

The local PostgreSQL URL and explicitly local password, immutable SQLite path, Inngest dev URL,
LangSmith project name, log level, and development environment are fixed in `compose.yaml`.
Application, prompt, and retrieval versions are code constants because they describe the checked-in
implementation rather than deployment configuration. Production deployments should inject their
managed database URL and Inngest credentials through their secret manager rather than reuse the
local Compose values.

Never put production credentials in source control or bake them into the image.

## Developer commands

Docker equivalents work on Windows without Make:

```bash
docker compose --env-file .env.local up --build
docker compose --env-file .env.local down
docker compose --env-file .env.local logs --follow app inngest postgres
docker compose --env-file .env.local run --rm migrate
docker compose --env-file .env.local run --rm app pytest
```

For a local Python development environment:

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Inspect the supplied knowledge database before changing retrieval:

```bash
uv run python scripts/inspect_database.py
```

## Reliability and security controls

- Slack `event_id` is unique in PostgreSQL and reused as the Inngest idempotency key.
- Inngest serializes work per `conversation_id` and caps shared model concurrency.
- Completed results are reusable; placeholder and response timestamps prevent duplicate messages.
- Knowledge access has no write-capable connection and no arbitrary SQL tool.
- Search values are parameterized; schema-derived identifiers are allowlisted and quoted.
- Search count, artifact batch size, evidence characters, retrieval rounds, and graph recursion are
  hard bounded.
- Retrieved text is explicitly treated as untrusted data, not instructions.
- No answer is generated without evidence; missing evidence produces a clear insufficiency response.
- Structured logs redact secret-like fields and carry request/run/conversation identifiers.
- Run records retain latency, action counts, token usage when returned by the model, sources, safe
  error codes, and version metadata.

## Observability

JSON logs cover HTTP correlation, retrieval duration/counts, Slack enqueue/delivery, and durable run
completion. PostgreSQL provides the application-level run ledger. The Inngest UI shows queue and
step retries. LangSmith traces carry the run ID, conversation ID, prompt/retrieval versions, model,
agent profile, environment, and app version. Offline experiment traces additionally carry the gold
dataset version and digest.

## Current limitations

- The supplied database has 250 artifacts and a standalone FTS5 index. Startup validates the exact
  documented schema and fails if it or FTS5 is unavailable. Lexical artifact retrieval is
  complemented by a parameterized relational account lookup for bounded cross-account questions;
  no arbitrary SQL is model-accessible.
- The official benchmark is intentionally only seven cases. Broader robustness coverage should be
  added as a separately reviewed suite after baseline experiments, not mixed into the gold set.
- Docker integration, PostgreSQL migrations, and live Inngest/Slack/OpenAI behavior require Docker
  and credentials; unit tests do not claim to validate those external systems.
- Retrieval is deliberately lexical until measured evaluation results justify hybrid/vector search.
