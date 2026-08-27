# Slack Q&A Agent

A Slack bot that answers questions from an immutable SQLite knowledge base using a bounded LangGraph workflow. Slack is the product interface; Docker Compose runs the local application, path-restricted Slack ingress, PostgreSQL, migrations, and Inngest.

This README is the complete local setup and run guide. Architecture, engineering decisions, and evaluation details are linked at the end.

## Prerequisites

- Git
- Docker with Compose v2 (Docker Desktop includes both)
- An OpenAI API key with access to `gpt-4.1-mini`, the code-defined production model
- A Slack workspace where you can create and install an app
- A free ngrok account and the [ngrok agent](https://ngrok.com/download)

You do not need to install Python, PostgreSQL, the Slack CLI, or the Inngest CLI for the recommended local path. You also do not run a Slack instance locally. Test through Slack's web client or desktop app.

## Included data

The complete upstream dataset bundle is included under `data/`. The runtime reads:

```text
data/synthetic_startup.sqlite
```

The application opens that database read-only, and Compose mounts it read-only.

## 1. Create and configure the Slack app

### Create a blank app

1. Open [Slack's app dashboard](https://api.slack.com/apps).
2. Select **Create New App**.
3. Select **Blank app**.
4. Name it e.g. `QA Agent`.
5. Select your workspace and create the app.

Do not create another Bolt codebase with the Slack CLI. This repository already contains the Bolt application.

### Add the bot scopes

Open **OAuth & Permissions**. Under **Scopes → Bot Token Scopes**, add exactly:

- [`app_mentions:read`](https://docs.slack.dev/reference/scopes/app_mentions.read/) allows Slack to deliver channel messages that explicitly mention `@QA Agent`.
- [`chat:write`](https://docs.slack.dev/reference/scopes/chat.write/) allows the bot to post and update its replies.

These scopes do not filter questions by subject. Any channel message that mentions `@QA Agent` is passed to the agent, which answers when the included knowledge database contains sufficient evidence.

No user-token scopes are required. The bot replies only in conversations where it has been invited. This integration is channel-mention driven and does not subscribe to direct-message events.

### Install the app and copy its credentials

1. On **OAuth & Permissions**, select **Install to Workspace** and approve the scopes.
2. Copy the **Bot User OAuth Token**, which starts with `xoxb-`.
3. Open **Basic Information → App Credentials**.
4. Reveal and copy the **Signing Secret**.

If you change scopes later, use **Reinstall to Workspace** before testing again.

## 2. Create `.env.local`

On Windows PowerShell:

```powershell
Copy-Item .env.example .env.local
```

On macOS or Linux:

```bash
cp .env.example .env.local
```

Fill in the OpenAI key and the two Slack credentials from the previous step:

```dotenv
OPENAI_API_KEY=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...

# Optional. Runtime tracing requires both LANGSMITH_TRACING=true and a LangSmith API key.
# The API key is also required by LangSmith sync, experiment, and augmentation commands.
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

LangSmith is not required to run or test the Slack bot. `.env.local` is ignored by Git and excluded
from Docker build contexts; never commit local credentials.

## 3. Start the local stack

On the first startup, or after changing application code, dependencies, or the Dockerfile, build the
application image and start the services in the background:

```bash
docker compose --env-file .env.local up -d --build
```

For normal subsequent startups, reuse the existing image:

```bash
docker compose --env-file .env.local up -d
```

Compose may still show a short cached build when the local image is missing, for example after an
image cleanup or an interrupted replacement. Lines marked `CACHED` do not repeat dependency
installation or compilation. `docker compose run` is for one-off commands; it does not start the
complete long-running stack.

Compose starts:

- the FastAPI and Slack Bolt application on port `8000`;
- a Slack-only ingress proxy on port `8001`;
- PostgreSQL on port `5432`;
- a one-shot Alembic and LangGraph checkpoint migration service;
- the Inngest dev server and UI on port `8288`.

The `migrate` container should finish with `Exited (0)`. It applies the schemas once and exits; that
does not mean the stack stopped. The `app`, `postgres`, `inngest`, and `slack-ingress` services
continue running. Check their current state with:

```bash
docker compose --env-file .env.local ps
```

Wait for the services to start, then verify:

- <http://localhost:8000/healthz> returns `{"status":"ok"}`;
- <http://localhost:8000/readyz> returns HTTP `200` and `"status":"ready"`;
- <http://localhost:8288> opens the Inngest development UI.

To follow the service logs:

```bash
docker compose --env-file .env.local logs --follow app slack-ingress migrate inngest postgres
```

## 4. Expose the Slack endpoint over HTTPS

Keep Docker running and open a second terminal. On a new ngrok installation, copy the authtoken from the [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken) and configure the agent once:

```bash
ngrok config add-authtoken <your-ngrok-authtoken>
```

Then start the tunnel:

```bash
ngrok http 8001
```

Port `8001` exposes only `POST /slack/events`. Keep ngrok running and copy the HTTPS forwarding URL
it prints. Do not tunnel port `8000`: that port also hosts local health and Inngest routes. Free
ngrok URLs may change after a restart; if that happens, update the Slack Request URL in the next
step.

## 5. Connect Slack's Events API

Return to the Slack app dashboard:

1. Open **Event Subscriptions**.
2. Turn **Enable Events** on.
3. Set **Request URL** to `https://<public-tunnel-host>/slack/events`.
4. Wait for Slack to mark the URL as verified.
5. Under **Subscribe to bot events**, add `app_mention`.
6. Save the changes.
7. Reinstall the app if Slack prompts you to apply permission changes.

Slack immediately sends a URL-verification challenge when the Request URL is saved. The application must be running and reachable through the tunnel at that point. Slack Bolt handles the challenge, [request-signature validation](https://docs.slack.dev/authentication/verifying-requests-from-slack/), and request-timestamp validation. Slack documents this flow in its [HTTP Request URL guide](https://docs.slack.dev/apis/events-api/using-http-request-urls/).

## 6. Invite and test the bot

Invite the app to a test channel using the channel integration menu or Slack's `/invite` command:

```text
/invite @QA Agent
```

Then mention it with a question:

```text
@QA Agent For Verdant Bay, what is the approved live patch window?
```

Expected behavior:

1. Slack delivers the mention to `/slack/events` and receives an acknowledgement after durable
   enqueueing succeeds.
2. The event appears in the Inngest UI at <http://localhost:8288>.
3. The bot posts a `Searching the knowledge base…` placeholder in the Slack thread.
4. The placeholder is updated with a grounded answer and source artifact IDs.

Every question, including a follow-up, must mention `@QA Agent`. Follow-ups in the same Slack thread reuse conversation state; different threads remain isolated.

## Troubleshooting local Slack delivery

If Slack cannot verify the Request URL:

- confirm the URL is HTTPS and ends with `/slack/events`;
- confirm <http://localhost:8000/healthz> works locally;
- confirm the tunnel is still running and forwards to port `8001`;
- inspect `docker compose --env-file .env.local logs app slack-ingress migrate inngest`;
- restart Compose after changing `.env.local`.

If Slack verifies the URL but mentions do not reach the bot:

- confirm `app_mention` is under **Subscribe to bot events**;
- confirm `app_mentions:read` and `chat:write` are bot-token scopes;
- reinstall the app after any scope change;
- invite the bot to the channel;
- confirm Slack still has the current tunnel URL.

## Direct agent diagnostic

This invokes the same transport-independent agent without sending a Slack message:

```bash
docker compose --env-file .env.local run --rm app python -m knowledge_assistant.cli ask "For Verdant Bay, what is the approved live patch window?"
```

Reuse multi-turn state by passing the same `--conversation-id` value to later calls. This command is a development diagnostic, not a second product interface or a fallback for Slack.

## Run the checks

With [uv](https://docs.astral.sh/uv/) installed locally:

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Or use the Docker validation stage:

```bash
docker build --target validation .
```

## Stop or reset

Stop the stack while keeping the local PostgreSQL volume:

```bash
docker compose --env-file .env.local down
```

Remove the stack and its local PostgreSQL data:

```bash
docker compose --env-file .env.local down --volumes
```

## Supporting documentation

- [Architecture and request lifecycle](docs/architecture.md)
- [Engineering decisions and tradeoffs](docs/decisions-and-tradeoffs.md)
- [Evaluations and LangSmith](docs/evaluations.md)
