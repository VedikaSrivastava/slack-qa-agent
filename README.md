# QA Agent

A Slack Q&A bot backed by the included read-only SQLite knowledge base. Docker Compose is the recommended reviewer path.

## Prerequisites

- Git
- Docker with Compose v2
- an OpenAI API key with access to `gpt-4.1-mini`
- a Slack workspace where you can create an app and use the Agent feature
- [ngrok](https://ngrok.com/download) and a free ngrok account

Python, PostgreSQL, the Slack CLI, and the Inngest CLI are not required on the host. The runtime uses `data/synthetic_startup.sqlite` and mounts it read-only.

## 1. Create and install the Slack app

Use [`slack-app-manifest.yaml`](slack-app-manifest.yaml) as the bootstrap manifest:

1. Open [Slack's app dashboard](https://api.slack.com/apps) and select **Create New App**.
2. Select **From an app manifest**, choose the workspace, select **YAML**, and paste the file.
3. Create the app, open **OAuth & Permissions**, and select **Install to Workspace**.
4. Copy the **Bot User OAuth Token** (`xoxb-...`).
5. Open **Basic Information > App Credentials** and copy the **Signing Secret**.

The bootstrap has the Agent feature and required scopes but no event subscriptions because the local HTTPS URL does not exist yet. A complete manifest is generated in step 5. If Slack displays **Reinstall to Workspace**, **Request to Reinstall**, or another authorization prompt after a manifest or scope change, complete it before testing.

## 2. Create `.env.local`

Windows PowerShell:

```powershell
Copy-Item .env.example .env.local
```

macOS or Linux:

```bash
cp .env.example .env.local
```

Set the required credentials:

```dotenv
OPENAI_API_KEY=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...

# Default: clear unmentioned follow-ups in an agent-owned thread may be handled.
SLACK_ROUTING_POLICY=agent_owned_thread_follow_ups

# Optional; LangSmith is not required for local use.
LANGSMITH_TRACING=false
LANGSMITH_API_KEY=
```

Set `SLACK_ROUTING_POLICY=explicit_mentions_only` to require `@QA Agent` on every turn.

## 3. Start the stack

If an earlier local checkout created the disposable PostgreSQL volume, reset it once:

```bash
docker compose --env-file .env.local down --volumes
```

Build and start:

```bash
docker compose --env-file .env.local up -d --build
```

Later starts can omit `--build`:

```bash
docker compose --env-file .env.local up -d
```

Check the services:

```bash
docker compose --env-file .env.local ps
```

The one-shot `migrate` service should finish with `Exited (0)`. The remaining services should keep running. Verify:

- <http://localhost:8000/healthz> returns `{"status":"ok"}`;
- <http://localhost:8000/readyz> returns HTTP `200` and `"status":"ready"`;
- <http://localhost:8288> opens the Inngest development UI.

Follow logs with:

```bash
docker compose --env-file .env.local logs --follow app slack-ingress migrate inngest postgres
```

## 4. Start the HTTPS tunnel

Configure ngrok (only once) if needed:

```bash
ngrok config add-authtoken <your-ngrok-authtoken>
```

Start the tunnel in a second terminal:

```bash
ngrok http 8001
```

Copy the HTTPS forwarding origin. Port `8001` exposes only `POST /slack/events`; do not tunnel port
`8000`. If the ngrok URL changes, regenerate and reapply the manifest in the next step.

## 5. Generate and apply the final manifest

Keep Compose and ngrok running. Generate the complete manifest with the HTTPS origin from ngrok.

Windows PowerShell:

```powershell
$publicBaseUrl = "https://<public-tunnel-host>"
$generatedManifest = docker compose --env-file .env.local exec -T app `
  python -m knowledge_assistant.integrations.slack.manifest $publicBaseUrl
$generatedManifest | Set-Clipboard
```

macOS or Linux:

```bash
public_base_url="https://<public-tunnel-host>"
docker compose --env-file .env.local exec -T app \
  python -m knowledge_assistant.integrations.slack.manifest "$public_base_url" \
  > slack-app-manifest.generated.yaml
```

Then:

1. Open **App Manifest** in the Slack app dashboard and select **YAML**.
2. Replace the bootstrap manifest with the generated output and save it.
3. Wait for Slack to verify `https://<public-tunnel-host>/slack/events`.
4. Confirm the bot events are `agent_session_stopped`, `app_mention`, `message.channels`, and
   `message.groups`.
5. Complete any reinstall or reauthorization prompt.

## 6. Invite and test QA Agent

Invite the app in each public or private channel where it should answer:

```text
/invite @QA Agent
```

Ask a question:

```text
@QA Agent For Verdant Bay, what is the approved live patch window?
```

See the [thread/session contract](docs/thread-and-session-model.md) for interaction behavior.

## Troubleshooting

If Slack cannot verify the Request URL:

- confirm the URL is HTTPS and ends with `/slack/events`;
- confirm <http://localhost:8000/healthz> works;
- confirm ngrok still forwards to port `8001`;
- regenerate the final manifest after a tunnel URL change;
- inspect `docker compose --env-file .env.local logs app slack-ingress migrate inngest`.

If mentions do not reach QA Agent:

- confirm the generated manifest, not the bootstrap, is currently applied;
- confirm `app_mention` is subscribed;
- reinstall after any scope or Agent-feature change;
- run `/invite @QA Agent` in the channel;
- restart Compose after changing `.env.local`.

If unmentioned follow-ups do not work:

- confirm `SLACK_ROUTING_POLICY=agent_owned_thread_follow_ups`;
- confirm `message.channels` is subscribed, plus `message.groups` and `groups:history` for private
  channels;
- begin the thread with an explicit mention and wait for its answer;
- inspect **Route Slack turn** in the Inngest UI; ambiguous messages intentionally stay silent.

If progress or Stop is missing:

- confirm the Slack Agent feature is enabled and the app has `assistant:write`;
- confirm `agent_session_stopped` is subscribed;
- inspect logs for `slack_stream_open_failed`; a stream-open failure still permits one final answer.

## Run checks

With [uv](https://docs.astral.sh/uv/) installed:

```bash
uv sync --frozen --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```

Or use Docker:

```bash
docker build --target validation .
```

## Stop or reset

```bash
# Stop while keeping PostgreSQL data.
docker compose --env-file .env.local down

# Remove the stack and local PostgreSQL data.
docker compose --env-file .env.local down --volumes
```

## Documentation

- [Architecture and request lifecycle](docs/architecture.md)
- [Slack thread and Agent Session model](docs/thread-and-session-model.md)
- [Engineering decisions and tradeoffs](docs/decisions-and-tradeoffs.md)
- [Implementation journal](docs/implementation-journal.md)
- [Evaluations and LangSmith](docs/evaluations.md)
