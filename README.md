# QA Agent

A Slack Q&A bot backed by the included read-only SQLite knowledge base. It supports grounded
multi-turn answers, optional source display, and bounded retrieval.

## Prerequisites

- Git
- Docker with Compose v2
- an OpenAI API key
- a Slack workspace where you can create an app and use the Agent feature
- [ngrok](https://ngrok.com/download) and a free ngrok account

Python, PostgreSQL, the Slack CLI, and the Inngest CLI are not required on the host. The runtime uses `data/synthetic_startup.sqlite` and mounts it read-only.

## 1. Create and install the Slack app

Use [`slack-app-manifest.yaml`](slack-app-manifest.yaml) as the bootstrap manifest:

1. Open [Slack's app dashboard](https://api.slack.com/apps) and select **Create New App**.
2. Select **From an app manifest**, choose the workspace, select **YAML**, and paste the file.
3. Create the app, open **OAuth & Permissions**, and select **Install to Workspace**.
4. Copy the **Bot User OAuth Token** (`xoxb-...`).
5. Open **App Settings > Basic Information > App Credentials** and copy the **Signing Secret**.

The bootstrap has the Agent feature and required scopes but no event subscriptions because the local HTTPS URL does not exist yet. A complete manifest is generated in step 6. Slack may show Agent View suggestions for `message.im` and `app_home_opened`; they are expected, because this integration supports invited channel mentions rather than direct messages. If Slack displays **Reinstall to Workspace**, **Request to Reinstall**, or another authorization prompt after a manifest or scope change, complete it before testing.

## 2. Create `.env.local`

```bash
cp .env.example .env.local
```

Set the required credentials:

```dotenv
OPENAI_API_KEY=...
SLACK_BOT_TOKEN=xoxb-...
SLACK_SIGNING_SECRET=...

# Optional stricter override; omit it to allow clear follow-ups in agent-owned threads.
# SLACK_ROUTING_POLICY=explicit_mentions_only
```

The code-reviewed default is `agent_owned_thread_follow_ups`. Set the optional override to
`explicit_mentions_only` to require `@QA Agent` on every turn.

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

## 4. Optional: enable Langfuse tracing

Langfuse tracing is optional. The QA agent runs normally without it. To inspect local traces, start
the self-hosted observability profile:

```bash
docker compose --env-file .env.local --profile observability up -d --build
```

Open <http://localhost:3000>, create a local Langfuse organisation and project, and add its public
and secret keys to `.env.local`:

```dotenv
LANGFUSE_PUBLIC_KEY=...
LANGFUSE_SECRET_KEY=...
```

Recreate the app container so it receives the new environment values:

```bash
docker compose --env-file .env.local up -d --force-recreate app
```

Langfuse records the LangGraph/LangChain execution tree, including model calls, retrieval steps,
latency, token usage, and configured run metadata. Local evaluation reports remain independent of
tracing and are written to `evals/reports/`; the offline evaluation harness deliberately disables
Langfuse even if its keys are present in `.env.local`.

If `LANGFUSE_BASE_URL` points to a hosted Langfuse service, trace payloads leave the application
environment. They may include user questions, prompt and model/tool inputs, retrieved evidence
snippets, generated answers, and run metadata. Enable hosted tracing only with explicit approval for
that data transfer and an accepted access, retention, and deletion policy. Never commit Langfuse
keys or expose private trace links.

## 5. Start the HTTPS tunnel

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

## 6. Generate and apply the final manifest

Keep Compose and ngrok running. Generate the complete manifest with the HTTPS origin from ngrok.

```bash
public_base_url="https://<public-tunnel-host>"
docker compose --env-file .env.local exec -T app \
  python -m knowledge_assistant.integrations.slack.manifest "$public_base_url" \
  > slack-app-manifest.generated.yaml
```

Then:

1. Open **App Manifest** in the Slack app dashboard and select **YAML**.
2. Replace the bootstrap manifest with the generated output and save it.
3. In Slack's `request_url` warning, select **Click here to verify** for `https://<public-tunnel-host>/slack/events`; once verification succeeds, Slack has applied the saved manifest changes.
4. Got to Event Subscriptions and confirm the bot events are `agent_session_stopped`, `app_mention`, `message.channels`, and
   `message.groups`.
5. Complete any reinstall or reauthorization prompt.

## 7. Invite and test QA Agent

Invite the app in each public or private channel where it should answer:

```text
/invite @QA Agent
```

and then ask a question:

```text
@QA Agent For Verdant Bay, what is the approved live patch window?
```

Replies in the same Slack thread continue the same agent conversation. Mention QA Agent again for
an explicit follow-up; clear unmentioned follow-ups are handled conservatively.

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

Run the official deterministic benchmark directly with `uv`. It uses an in-process checkpointer and
does not require Docker, Slack credentials, Langfuse, PostgreSQL, or a hosted evaluation service:

```bash
uv run python -m knowledge_assistant.evals run --suite full \
  --profile balanced-gpt-4.1-mini --env-file .env.local \
  --output evals/reports/full-balanced-gpt-4.1-mini.json
```

This live command sends the seven questions, retrieved evidence, and generated answers to OpenAI.
Run it only with authorization for that transfer. Offline evaluation deliberately disables
Langfuse. Exit code zero means the run completed and wrote a valid report; it does not mean every
answer or deterministic gate passed.

The final recorded deterministic snapshot is
[`takehome-agent-p19-r12-e13-final-20260829-01.json`](evals/reports/takehome-agent-p19-r12-e13-final-20260829-01.json).
It uses prompt `v19`, retrieval `v12`, and evaluation protocol `v13`: 6/7 strict-contract cases,
3/4 applicable exact-content cases, 7/7 evidence and operation contracts, and
`semantic_quality: not_judged`. Manual assignment-reference review found 5/7 fully complete,
reference-agreeing answers, so the 7/7 target was not met. See
[Evaluation findings](docs/evaluation-findings.md) for the case review and metric caveats. Matching
p19/r12 derived and multi-turn regression runs have not been measured.

### Optional semantic evaluation: authorization required

A judge run additionally sends generated answers and reference answers to the judge model. Run it
only with explicit authorization for that expanded transfer, and acknowledge the boundary in the
command:

```bash
uv run python -m knowledge_assistant.evals judge --label finalists --suite full \
  --profiles balanced-gpt-4.1-mini --judge-model gpt-5 \
  --env-file .env.local --confirm-data-transfer
```

Judge output is a candidate-defined diagnostic, not an assignment pass threshold. Future judge
protocol-v5 reports record the explicit transfer acknowledgement and combine semantic answer
quality with the strict deterministic contract in `task_quality_passed`. The default take-home
evidence is the official deterministic run plus manual review. Derived and multi-turn suites are
candidate-authored regression checks, not held-out proof. See
[Evaluation strategy](docs/evaluations.md) for the complete policy and production evaluation plan.

Every `evals/reports/candidate-*` directory lacks documented authorization metadata and is
unaccepted. Preserve these artifacts only for audit; do not quote their rates or use them as
submission evidence.

When dependencies change, regenerate the lockfile at the repository root:

```bash
uv lock
uv sync --frozen --all-groups
```

## Stop or reset

```bash
# Stop while keeping PostgreSQL data.
docker compose --env-file .env.local down

# Remove the stack and local PostgreSQL data.
docker compose --env-file .env.local down --volumes
```
