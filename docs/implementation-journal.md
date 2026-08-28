# Implementation journal

Last updated: 2026-08-27

This is a factual engineering record of the implementation work, investigations, rejected
approaches, failure modes, and remaining validation gaps. It is intentionally more detailed than
the reviewer setup guide so development context is not lost.

## Current product contract

The current repository implements a single-workspace Slack knowledge-QA agent with these product
boundaries:

- public and private channel threads are supported; direct-message events are not subscribed;
- an explicit mention in a root message establishes one agent-owned Slack thread;
- the root thread identity is also the LangGraph conversation and Slack Agent Session identity;
- a mention inside that thread is a new run in the same conversation, never a nested thread;
- clear unmentioned follow-ups may be handled after a conservative responder judgment;
- any channel participant may ask a follow-up under the current shared-corpus authorization
  assumption;
- the agent exposes genuine code-owned progress stages through Slack's native task stream;
- no private reasoning, evidence text, or unverified draft is streamed;
- the verified answer appears as a coherent final chunk, with ordered continuation messages only
  when Slack's size limit requires them;
- Agent Session Stop is supported through a durable, race-safe cooperative cancellation path;
- greetings, unclear messages, and out-of-scope requests have explicit terminal behavior;
- the supplied SQLite corpus is immutable and read-only;
- PostgreSQL stores application run/delivery truth and LangGraph checkpoints;
- Inngest owns background retries, scheduling, concurrency limits, and side-effect sequencing.

## Slack platform investigation

The original assignment language predates current Slack Agent capabilities, so the Slack portion
was rechecked against the current official documentation rather than treating the assignment's
platform limitation as permanent.

The current platform provides:

- [`chat.startStream`](https://docs.slack.dev/reference/methods/chat.startStream/) to create a
  streaming message;
- [`chat.appendStream`](https://docs.slack.dev/reference/methods/chat.appendStream/) to append
  markdown or structured task/plan chunks;
- [`chat.stopStream`](https://docs.slack.dev/reference/methods/chat.stopStream/) to finalize it;
- [Agent Sessions](https://docs.slack.dev/ai/agent-sessions/) scoped to a channel thread, with
  `processing`, `active`, `suspended`, and `closed` states;
- `agent_session_stopped` for the native Stop control;
- the current [`agent_view` manifest feature](https://docs.slack.dev/reference/app-manifest/),
  which replaces the older Assistant presentation for new apps;
- `assistant:write` to declare the app as an Agent and `chat:write` for current session status and
  streaming operations.

The implementation uses the current `agents.sessions.setStatus` API. It does not use the legacy
`assistant.threads.*` methods or Bolt's older Assistant class.

## Progress and final-answer evolution

### Editable placeholder

An early approach posted a progress/placeholder Slack message and repeatedly updated it before
replacing it with the answer. That path had two product problems:

1. it looked cosmetic when the text was not tied to actual workflow boundaries;
2. replacing a posted message could surface Slack's edited-message treatment, which makes the final
   response feel less native and less trustworthy.

That approach has been removed. There is no progress-placeholder setting, placeholder timestamp,
repository API, or schema column in the clean baseline.

### Character-, word-, or token-level draft streaming

Streaming generated prose immediately would feel responsive, but the agent's core safety contract
is that factual claims are grounded and cited. Generation is followed by deterministic citation
checks and a model grounding audit, with one bounded repair opportunity. Exposing the draft before
those checks would let users read claims that may later be removed or corrected.

The selected behavior is therefore:

1. stream real, sanitized task state while the workflow runs;
2. keep draft prose private;
3. finish verification;
4. send the complete canonical answer through `chat.stopStream`;
5. split only responses above the Slack stream-markdown limit, preserving deterministic order.

This is block-level final delivery rather than an animation of characters or words. The progress is
still genuine because every label comes from an actual graph stage emitted by the processor.

### Operational stream failure

Native streaming is required by the manifest and runtime; feature flags for disabling Agent
Sessions or choosing the old progress surface were removed. A transport failure still needs a
production-safe outcome, so degradation now means:

- do not create an editable substitute progress message;
- suppress later task updates because there is no trustworthy stream identity;
- keep the native thread session loader best-effort;
- continue bounded reasoning;
- post the verified final answer once with a deterministic `client_msg_id`;
- make session terminal status a required delivery effect.

`chat.startStream` has no application idempotency key. SDK transport retries are disabled so a
hidden retry cannot create two streams before either timestamp reaches PostgreSQL. Inngest owns the
visible durable retry policy. If the SDK returns an ambiguous transport failure to application code,
the application degrades progress rather than deliberately calling start again.

The known-success path is divided into three durable steps rather than combining two systems in one
callback:

1. a database-only step claims the progress surface;
2. one Slack-only step calls `chat.startStream` and returns the timestamp;
3. a database acknowledgement step persists that exact timestamp and reconciles concurrent Stop.

Inngest stores the second step's output before running the third. A transient PostgreSQL failure in
step 3 therefore retries with the same Slack timestamp and cannot call `chat.startStream` again. If
the start response itself is returned as an error, the remote result is unknowable because Slack
exposes no idempotency key or lookup-by-client-ID for stream creation; application code suppresses
later progress and uses the immutable final-post path rather than guessing.

A harder failure exists below that handled-error path: the process can exit after Slack successfully
creates the stream but before Inngest durably stores the step output. Inngest may then retry the
non-idempotent start step without knowing the first timestamp. That can create a duplicate progress
surface and leave the first one orphaned because the application has no identity with which to stop
it. The three-step protocol prevents a later database-acknowledgement retry from repeating start,
but it does not provide exactly-once progress creation across this process-crash window. Final-answer
delivery remains a separate ledger-backed protocol.

An ambiguous `chat.stopStream` is handled differently because the stream timestamp and canonical
answer are already durable. Every canonical stop carries message metadata containing the run,
delivery kind, and answer hash. A retry sends the identical final chunks to the same stream. If
Slack reports `message_not_in_streaming_state`, that error alone is not delivery proof: a user may
have clicked Stop before finalization. The publisher reads the message back and accepts it only when
the canonical metadata matches. Slack history can briefly expose the pre-stop form of the same
message, so one metadata-free observation is not enough: metadata must be absent from the exact
message on every bounded poll before the publisher posts the answer or cancellation confirmation as
an immutable message with a deterministic `client_msg_id`. A later match, failed read, mixed result,
or response without the exact timestamp is `unknown`, not absence, and raises into Inngest's durable
retry instead of risking a duplicate. The implementation never calls `chat.update` to rewrite a
finalized response.

## Sanitized progress boundary

The processor emits a typed `ProgressEvent` with only:

- stable run and event IDs;
- a monotonic sequence;
- one code-owned stage enum;
- an optional bounded retrieval-round number.

The stream opens with the neutral and truthful `Understanding the request` task; greetings,
clarification requests, and scope responses never claim to be researching. The remaining available
labels are searching company knowledge, reviewing evidence, drafting, verifying, and tightening.
Prompt text, model scratchpads, retrieved content, search results,
customer data, SQL, and unverified answer text cannot cross this model boundary. Slack publishing
maps the enum to a fixed task update; it never accepts arbitrary model-generated progress prose.

## Thread and session investigation

Slack replies carry the timestamp of the root thread. There is no nested reply-thread identity, so
the stable mapping is:

```text
workspace + channel + root thread timestamp
    = application conversation_id
    = LangGraph thread_id
    = Slack thread-scoped Agent Session
```

An explicit mention in an existing thread therefore starts another run, not another conversation.
A new root mention starts a different conversation. Full cases, participant rules, rapid-turn
ordering, and Stop targeting are recorded in
[Slack thread and Agent Session model](thread-and-session-model.md).

## Follow-up responder judgment

Automatically answering every message in a shared thread would cause the bot to interrupt human
conversation. Requiring a mention forever makes multi-turn use unnatural. The selected middle path
is:

1. an explicit mention and every ordinary follow-up candidate enter the same durable turn queue;
2. a delivered answer to an explicit mention establishes agent ownership of that thread;
3. when an ordinary candidate becomes the causal head, PostgreSQL confirms ownership of the exact
   workspace/channel/root thread;
4. the router invokes a structured classifier that returns only `respond`, `stay_silent`, or
   `uncertain`;
5. only `respond` creates and links a question run; explicit mentions bypass classification;
6. failures and uncertainty terminate the candidate as suppressed without a run.

The classifier is deliberately behind Inngest instead of Slack's acknowledgement path. Explicit
mentions do not pay for this classifier. The `explicit_mentions_only` configuration remains a
product-policy switch, not a compatibility path.

## Greeting, ambiguity, and scope investigation

Sending `hey` through search and then returning an insufficient-evidence message is technically
grounded but poor conversation behavior. Guessing what an unclear `what about it?` refers to is
worse. Treating a weather, creative-writing, or action request as a failed knowledge search also
misstates the bot's responsibility.

The graph now has a typed `QuestionDisposition`:

- `knowledge_question` enters retrieval and grounding;
- `greeting` returns a short code-owned response;
- `needs_clarification` asks one normalized follow-up question;
- `out_of_scope` explains the grounded knowledge-QA boundary and invites a relevant question.

Whole-message greetings are recognized deterministically and use zero model/tool actions. For all
other messages, disposition was added to the existing structured retrieval plan, so a normal
knowledge question does not incur an additional model call. Conversation resolution happens first;
an ambiguous follow-up that recent thread history can resolve still proceeds normally.

Direct conversational outcomes skip evidence and citation requirements because they make no corpus
claim. A clarification outcome is persisted in the normal `AgentResponse` snapshot and causes Slack
delivery to set the session to `suspended`, explicitly representing that work is waiting on the
user.

## Why Inngest remains in the architecture

The model workflow and Slack delivery can exceed Slack's acknowledgement deadline and can fail at
different boundaries. Inngest is useful here because it gives one owner to:

- durable background invocation after the short signed-event handoff;
- event idempotency;
- bounded retries;
- per-conversation and global model concurrency;
- timeouts;
- cancellation triggers;
- ordered, separately acknowledged delivery steps;
- a visible local execution timeline.

It is not used as a wrapper around every SQLite/PostgreSQL read and write, and it does not replace
the database. LangGraph still owns bounded semantic transitions; PostgreSQL still owns durable
application truth.

### Shared causal turn-router correction

An earlier iteration sent explicit mentions directly toward question processing while ordinary
replies passed through a separate follow-up router. It also created the queued run in PostgreSQL
before sending a new question event. That split left two production problems:

- mentions and follow-ups had no single causal head, so an invocation from one function could
  overtake work admitted through the other;
- a process failure after the database commit but before the new Inngest send could leave an
  `agent_runs` row that no durable function owned.

Keyed Inngest concurrency alone was not sufficient. It is capacity and contention control, and the
runtime can release that capacity during durable sleeps and child-function invocations. The
correction made PostgreSQL `slack_turns` the shared causal queue for both input event types:

1. `route-slack-turn` inserts or reuses the exact event identity;
2. a transaction-scoped advisory lock serializes queue mutations for the conversation;
3. pending rows are compared using a numeric Slack timestamp, then creation time and event ID;
4. a partial unique index permits only one `processing` owner;
5. a duplicate event ID must match workspace, channel, user, message timestamp, root timestamp, and
   turn kind exactly;
6. ordinary candidates invoke the separately model-bounded classifier only after they own the head;
   explicit mentions bypass it;
7. only an accepted turn creates `agent_runs`, and run creation plus immutable turn linking happen
   in one PostgreSQL transaction;
8. the router invokes progress and then primary processing with `step.invoke`, verifies the run is
   terminal, and only then completes the linked turn as `routed`.

This removes the normal database-to-new-event dual write. The ingress sends only the original
stable event into an already durable router; all later work is a child invocation. Slack parsing
derives UUIDv5 run/candidate identities from workspace, Slack event ID, and purpose, while dispatch
uses stable prefixed Inngest event IDs. Slack retries therefore reproduce the same identities instead
of forking work.

Known later turns retry with durable 15-second sleeps, bounded to 240 attempts (about one hour).
Ordering is exact among rows known at claim time. A previously unseen older Slack event cannot
preempt a turn that is already `processing`; it waits behind that lease, then participates in the
remaining numeric order. Retroactive preemption was rejected because it would create two owners for
one conversation.

### Current function inventory

1. **Route Slack turn** owns the shared causal queue, optional responder decision, atomic run link,
   ordered child invocations, and terminal turn transition.
2. **Classify Slack follow-up** is invoked only for ordinary candidates and shares the global model
   concurrency bound; explicit mentions never invoke it.
3. **Initialize Slack progress** has independent Slack capacity so the loader opens before model
   work. It uses the same three-step database claim, remote start, and acknowledgement protocol as
   the primary backstop.
4. **Process Slack question** claims the run, ensures progress, runs/resumes the graph, persists the
   immutable result before delivery, checks cancellation, installs the delivery manifest, claims
   delivery, publishes ordered parts, completes delivery, and then marks the run successful.
5. **Clean up failed Slack turn** releases an unlinked failed turn or waits for a linked run to become
   terminal before completing that turn. It handles router failure and cancellation/timeout events;
   blocked cleanup retries use new durable claim steps so a predecessor's completion is observable.
6. **Clean up failed Slack question** reconstructs and resumes a persisted verified answer before
   considering a generic error, then reconciles complete/partial delivery and the Slack session
   without leaking provider errors.
7. **Clean up cancelled Slack question** distinguishes accepted user Stop from
   administrative/timeout cancellation and reaches the correct terminal user surface.
8. **Finalize an unreconciled Slack run** is the final dead-letter boundary for exhausted turn,
   question, or cancellation cleanup, including cleanup timeout/cancellation events. For a linked
   poisoned turn, it terminalizes or abandons the run first and then completes the turn so the
   conversation queue is released.

### Progress concurrency backstop

The causal router now invokes progress before primary processing, but independent function retries
still cannot act as a transactional lock over Slack. The database remains the progress correctness
boundary:

- select the oldest active run in the conversation under lock;
- allow only that run to move out of `not_started`;
- enforce one active progress owner with a partial unique index;
- let later initializers defer without side effects.

This resolved the risk of competing native progress displays in one Slack session.

## Stop and cancellation evolution

Several tempting shortcuts were rejected:

- selecting the newest active run for every Stop could let a delayed event cancel a later turn;
- emitting only an Inngest cancel event would lose intent if the handoff failed;
- trusting Inngest cancellation alone would not settle the race with an in-flight Slack write;
- cancelling directly in the signed-event handler would put network/model latency in the
  acknowledgement path;
- failing to persist a stream timestamp returned while Stop wins could leave an orphan loader.

The implemented protocol is:

1. validate and normalize the Stop event;
2. atomically insert an immutable event claim;
3. select the causal run using thread identity, reported stream identities, and message/event time;
4. lock the selected run and compete with final delivery on that row;
5. persist cancellation intent and the immutable accepted/rejected result;
6. emit an Inngest cancellation event only after the database decision;
7. check persisted cancellation between stages and before delivery;
8. finalize from either the primary path or the cleanup function;
9. reuse the stored outcome for Slack retries.

If a stream start returns after cancellation has won, the acknowledgement path persists the
timestamp and closes that late progress surface without terminal text. It never competes to publish
the Stop message; cancellation finalization is the single owner of the visible confirmation. This
removed a race in which cleanup could post a fallback while the delayed initializer finalized the
stream with the same notice.

Cancellation is cooperative. It cannot terminate an already-executing provider HTTP request, but a
result that finishes after cancellation won is withheld.

Slack stops the reported streaming messages before delivering `agent_session_stopped`, while the
Agent Session itself remains `processing` until the app changes it. This creates an important race:
the database may reject cancellation because final delivery already owns the run, even though the
user's click already stopped the Slack stream. Consequently, `message_not_in_streaming_state` never
acknowledges answer delivery on its own. Canonical metadata read-back distinguishes a successfully
completed prior stop from a user-stopped progress-only message; the latter receives the final answer
through the deterministic immutable-post fallback. Accepted cancellation follows the same rule so
the user always receives a visible stopped confirmation, and session status is explicitly restored
to `active`. A delayed progress initializer rechecks cancellation, response identity, and delivery
status before any `processing` status write, so it cannot reactivate the loader after cleanup has
completed.

Session status is not a generic cleanup reset. A rejected or stale Stop, a Stop with no matched run,
and cleanup that observes an already-succeeded run perform no session-status write. The successful
delivery/finalization path owns whether its terminal state is `active` or, for a clarification,
`suspended`. This prevents delayed Stop cleanup from overwriting a valid suspended clarification or
the state established by a later turn.

## Ordered long-answer delivery

Slack's stream markdown input has a 12,000-character bound. The canonical formatted response is
split at readable boundaries into one stream-finalizing part plus zero or more continuation posts.
The delivery manifest stores a version and ordered SHA-256 hashes before any final write.

Each part has its own durable Inngest step and PostgreSQL acknowledgement. A retry returns the
acknowledged Slack timestamp rather than posting again. Continuations use deterministic message
IDs. A later part cannot publish before every earlier part is acknowledged. If all parts reached
Slack but the last database transition was lost, cleanup promotes the run without reposting. If
only a prefix was acknowledged, cleanup preserves that useful answer and appends one deterministic
incomplete-delivery notice.

## Signed ingress and acknowledgement path corrections

Slack Bolt owns signature and request-timestamp validation. The public proxy exposes only the exact
`POST /slack/events` route; health, readiness, and the unsigned local Inngest endpoint remain on the
private application port.

A previous authorization shape could invoke Slack `auth.test` while processing a request. That
network dependency was moved to application startup. A validated single-workspace authorization
result is cached, and an event from a different workspace is rejected without a network call.
The same startup response now validates all required installed scopes so a manifest update without
the required reinstall cannot produce a healthy-looking but unusable runtime.

Mention normalization also uses that cached bot user ID. Only the exact authenticated bot token
`<@BOT_ID>` is removed from an `app_mention`; a question such as “What is documented for
`<@OTHER_USER>`?” preserves the referenced person instead of stripping every Slack user mention.

Every routed Slack event uses a purpose-specific UUIDv5 derived from workspace and Slack event ID.
This yields stable explicit-question run IDs, follow-up candidate/run IDs, and empty-mention response
IDs. Dispatch uses a stable prefixed Inngest event ID as a second deduplication boundary, so retries
reproduce the same identities instead of creating parallel work. An explicit mention with no question
receives a small help message within Slack's acknowledgement deadline.

## Persistence and migration reset

This repository has not been deployed or submitted, so retaining five development-only Alembic
revisions would create noise rather than compatibility value. They were replaced with one clean
`0001` baseline and the old local PostgreSQL/checkpoint state is intentionally disposable.

The application schema uses one flat `agent_runs` aggregate row for fields that participate in the
same lifecycle, Stop, stream, delivery, and completion state machine. Keeping those values together
allows one row lock to linearize Stop versus final delivery and terminal run transitions. Repeated
or independently queryable facts are normalized into:

- `slack_turns`, one immutable Slack input event and its causal-queue state;
- `run_sources`, one cited artifact per run;
- `run_delivery_parts`, one immutable ordered answer part per run;
- `slack_stop_events`, one idempotent Stop-event outcome;
- `slack_stopped_streams`, one ordered streaming-message identity reported by a Stop event.

The final `result_json` is a bounded, typed immutable replay snapshot used to avoid repeating model
work after a lost durable-step acknowledgement. It is not an arbitrary document store or query
surface. Database constraints cover enum domains, nonnegative metrics, stream/timestamp coherence,
manifest pairs, terminal timestamps, source ranks, part hashes, cancellation state, and unique
identities. The Slack timestamp list was deliberately normalized rather than stored as a JSON array;
the typed result snapshot is the only intentional JSON exception because it is read and written as
one immutable replay value, never filtered or joined as relational data.

Future schema changes after the baseline is shared should be new forward migrations. For this local
reset, reviewers/developers with an old Compose volume must remove it before startup:

```bash
docker compose --env-file .env.local down --volumes
```

## Database roles and RLS assessment

The supplied knowledge data has no tenant, customer-permission, or role mapping, and the
application-owned PostgreSQL tables are accessed only by the trusted service repository. There is
no end-user SQL/API query surface and no authorization principal column on which to build a
meaningful row policy.

RLS was therefore not enabled just to create the appearance of isolation. A policy without a real
tenant key would either deny legitimate service work or be bypassed by the same service role, which
does not solve the product authorization question.

The production hardening path is:

1. use a dedicated migration owner and a separate least-privilege runtime role;
2. revoke broad schema/table creation rights outside the migration path;
3. grant the runtime only the exact table and sequence operations it needs;
4. if the product introduces tenants or permission scopes, add the explicit authorization key to
   all relevant retrieval and operational data;
5. set request-scoped authorization context transactionally;
6. enable and force tested RLS policies, including cross-tenant negative tests;
7. include the same authorization dimension in retrieval, checkpoints, delivery, evaluations, and
   any cache key.

## Authorization assumption

Until the assignment owner specifies otherwise, the supplied SQLite file is treated as one
internal corpus. An eligible non-guest workspace member may query it only through a channel to
which they and the bot have access. This is channel membership control, not customer-level data
authorization. Slack's current Agent guidance says workspace guests cannot access apps with the
Agents feature enabled; guests are therefore not included in the participant assumption. The bot
must not claim to provide tenant isolation that the input schema cannot represent.

## Slack-derived state retention investigation

Slack's current
[Agent data-retention guidance](https://docs.slack.dev/ai/developing-agents/#data-retention)
recommends keeping metadata and retrieving Slack content at use time rather than storing Slack
data. The implementation was audited against that guidance instead of equating bounded prompt
history with retention compliance.

The current local state footprint is:

- application tables retain workspace, channel, user, message, thread, stream, and event identifiers,
  plus a typed final-result snapshot and delivery metadata;
- LangGraph checkpoints retain the current question, standalone rewrite, bounded user-visible turn
  history, evidence, intermediate draft, and final answer needed for resume and multi-turn behavior;
- the local Inngest development server retains event payloads and execution history used for durable
  dispatch and replay;
- explicitly enabled LangSmith tracing can create a separate remote copy of graph/model inputs and
  outputs, subject to that project's retention controls;
- the Slack messages themselves remain governed by the workspace and are not removed by deleting a
  local database volume.

No application TTL, Slack-message-deletion synchronizer, subject-erasure endpoint, workspace purge,
or backup-deletion protocol currently exists. `docker compose down --volumes` is useful for resetting
disposable local PostgreSQL state and removes the development containers, but it is not an
end-to-end deletion guarantee and cannot erase Slack or remote tracing data.

Several shortcuts were rejected as false assurances:

- limiting history to a few turns controls model context and cost but leaves checkpoint revisions;
- deleting only `agent_runs` would miss checkpoints, Inngest history, traces, caches, and backups;
- choosing an arbitrary TTL without product, legal, support, and incident-response requirements
  would create policy by accident;
- adding RLS would govern who can read retained rows, not when those rows must be deleted.

Before a production launch, the product must define data categories, purpose, retention periods,
erasure subjects, legal/audit holds, and deletion triggers. Implementation then needs to minimize raw
Slack content, fetch it in real time where practical, and propagate message deletion, user/workspace
erasure, and app-uninstallation requests through every durable store and disaster-recovery copy.
Tests must prove both deletion completion and authorization isolation; operational monitoring must
surface failed or delayed purges without logging the content being erased.

## Answer-cache assessment

Caching an identical question could reduce repeated model usage, but repeated text is not a safe
key by itself. The answer can depend on authorization, conversation history, corpus snapshot,
prompt/retrieval/model versions, source validity, and formatting behavior. A naive global cache
could leak data or return stale answers.

Caching was therefore deferred until measurements show a repeated-query bottleneck. A production
experiment would first define:

- workspace/tenant/permission scope;
- normalized standalone question and relevant bounded context;
- immutable corpus snapshot or invalidation version;
- model, prompt, retrieval, and answer-format versions;
- TTL and explicit invalidation;
- encryption and retention;
- source revalidation;
- stampede control and safe failure behavior;
- hit rate, latency, token savings, staleness, and authorization-isolation tests.

This is an optimization candidate, not a correctness dependency.

## Runtime readiness and Inngest startup investigation

The earlier readiness response labeled the agent, Slack, and Inngest as `configured`. That described
settings, not service availability, and could be misread as an operational probe. The application
endpoint now reports only the dependencies it directly checks: the immutable knowledge file and a
PostgreSQL connection. Slack authentication is validated during application startup, while OpenAI,
Slack API, and Inngest failures remain external runtime failures rather than fabricated readiness
states.

For the local stack, Inngest still needs the running application URL to discover functions, so making
the application wait on Inngest while Inngest waited on the application would create a dependency
cycle. The useful gate is the path that admits Slack traffic. The `slack-ingress` service therefore
waits for both application health and a real Inngest health check before exposing the signed Events
API route.

The pinned Inngest image already contains the official
[`inngest alpha doctor healthcheck`](https://www.inngest.com/docs/self-hosting#docker-compose-example)
command, so no `curl`, package installation, or additional health-probe image was needed. Compose
retries that command during startup. This removes the configuration-equals-readiness claim without
adding a dependency. It remains a startup ordering guard: Compose does not withdraw an already
running ingress container merely because Inngest later becomes unhealthy, so dispatch failures must
still fail safely and rely on Slack delivery retries and operational alerting.

## Reviewer setup improvements

Manual Slack setup instructions are easy to drift. The first manifest draft attempted to keep the
tunnel-specific Request URL manual while still declaring all four `bot_events`. That combination is
not an importable manifest: Slack requires `settings.event_subscriptions` to include a
`request_url` unless Socket Mode is enabled, and the URL must be verified before event delivery can
work. This repository intentionally uses request-based Events API delivery, and a local reviewer's
ngrok host cannot be known in advance.

The following alternatives were considered:

- checking in a fake Request URL, rejected because it cannot pass the URL-verification step and
  leaves reviewers with configuration that looks complete but cannot receive events;
- enabling Socket Mode only to make bootstrap import succeed, rejected because it would contradict
  the path-restricted HTTP ingress used by the application and deployment;
- declaring no manifest and asking reviewers to configure every scope, feature, and event manually,
  rejected because it is slow and prone to drift;
- keeping scopes in a bootstrap manifest but manually adding the events later, improved by generating
  the entire final manifest so the event names and Request URL stay one deterministic contract.

The resolved setup is deliberately two-stage:

1. `slack-app-manifest.yaml` is a valid bootstrap manifest containing the current Agent feature,
   bot identity, and scopes, but no `event_subscriptions` block.
2. Once the application, restricted ingress, and HTTPS tunnel are running, the checked-in
   `knowledge_assistant.integrations.slack.manifest` CLI validates the public HTTPS origin and emits
   a complete manifest with `<origin>/slack/events` and the exact four events.
3. The reviewer replaces the bootstrap manifest with that generated output. Saving it performs
   Slack's normal URL-verification challenge against the running application.

The generator works from the installed application package, so the recommended Docker-only setup
does not add Python as a host prerequisite. Its stdout mode supports PowerShell clipboard capture
and shell redirection; `--output` writes UTF-8 YAML for contributors using the local `uv`
environment. It rejects non-HTTPS schemes, credentials, query strings, fragments, path-prefixed
base URLs, localhost, and malformed ports before producing configuration. Unit contracts assert
that the checked-in bootstrap remains text-equivalent to the packaged bootstrap, contains no
subscriptions, and that the generated manifest contains the expected Request URL and each required
event exactly once.

The bootstrap manifest is now the source of truth for:

- current `agent_view` declaration;
- bot identity;
- `app_mentions:read`, `assistant:write`, `channels:history`, `chat:write`, and `groups:history`;
- HTTP Events API transport rather than Socket Mode.

The generated final manifest adds `app_mention`, `message.channels`, `message.groups`, and
`agent_session_stopped` together with the tunnel-specific Request URL. README setup explicitly calls
out Slack's reinstall/reauthorization prompt after scope or Agent feature changes and
`/invite @QA Agent` for channel membership. The relevant Slack platform references are the
[app manifest event-subscription schema](https://docs.slack.dev/reference/app-manifest/#event_subscriptions)
and [HTTP Request URL flow](https://docs.slack.dev/apis/events-api/using-http-request-urls/).

A correct checked-in manifest does not prove that an existing installation granted its latest
scopes. Startup therefore reads the case-insensitive `x-oauth-scopes` response header from the one
cached `auth.test` call and fails before readiness if any of the five runtime scopes is absent. The
error lists only missing scope names and tells the operator to update the manifest and reinstall; it
never logs or echoes the bot token. This turns a first-user stream/history failure into a setup-time
diagnostic.

## Approaches considered and outcome

| Consideration | Outcome | Reason |
| --- | --- | --- |
| Editable placeholder with repeated updates | Removed | Non-native final appearance and can look like simulated progress. |
| Stream draft prose token by token | Rejected | Users could see claims before grounding and repair. |
| Native task stages plus verified final chunk | Implemented | Genuine progress without exposing reasoning or unverified content. |
| Optional non-Agent mode | Removed | Manifest and runtime now require the intended Agent UX. |
| One immutable final post after stream-open failure | Implemented | Preserves answer reliability without pretending to stream. |
| Automatic SDK retry of `chat.startStream` | Disabled | The method lacks an application idempotency key. |
| Exactly-once `chat.startStream` across worker death | Not available | Slack cannot recover a stream by a caller-supplied idempotency key, so death before Inngest stores the timestamp can duplicate/orphan progress. |
| `chat.update` after ambiguous final stop | Removed | Retrying identical stop chunks preserves canonical content without an edited rewrite. |
| Progress only inside the model-constrained function | Expanded | A separate initializer makes the loader responsive under model load. |
| Inngest concurrency as the only lock | Rejected | Cross-function ordering needs a database invariant. |
| Cancel whichever run is newest | Rejected | Delayed Stop could cancel the wrong turn. |
| Persist Stop outcome before cancellation handoff | Implemented | Makes retries and handoff loss safe. |
| Run Slack authorization test per event | Removed | Network latency does not belong in the acknowledgement path. |
| Respond to every message in an owned thread | Rejected | The bot would interrupt human discussion. |
| Require mention for every turn | Optional policy | Safe but less natural; default uses a conservative responder judge. |
| Search on greetings or unclear questions | Removed | Creates irrelevant retrieval and misleading evidence failures. |
| Add a separate intake model call | Rejected | Existing structured planning can own disposition at no extra normal-call cost. |
| Preserve five pre-release migrations | Removed | No deployed state requires compatibility; one baseline is clearer. |
| Add RLS without a tenant/principal key | Rejected | It would be cosmetic rather than an authorization control. |
| Global generated-answer cache | Deferred | No measured need and key/invalidation/authorization are not yet defined. |
| Add a vector database immediately | Deferred | Existing FTS5 and structured lookup should be evaluated before adding an index pipeline. |

## Validation record and remaining live checks

The implementation is covered by deterministic unit tests for parsing, routing, workflow
dispositions, retrieval limits, progress ordering, Stop races, delivery replay, long answers,
configuration, migrations, and readiness. The supplied SQLite integration test remains separate and
uses the included database.

Live Slack behavior cannot be proven by mocked SDK tests. Before treating a deployment as complete,
exercise these cases in the target workspace:

1. manifest import and reinstall authorization;
2. `/invite` in one public and one private channel;
3. root mention, explicit mentioned follow-up, and clear unmentioned follow-up;
4. greeting, unclear initial request, context-resolved follow-up, and out-of-scope request;
5. native stage progression and verified final chunk;
6. Stop before model work, during a model request, and immediately before delivery;
7. two rapid questions in one thread and simultaneous questions in different threads;
8. an answer above the stream limit;
9. an induced Slack stream-open failure;
10. Slack event retries and Inngest step retries;
11. reset/upgrade from an empty PostgreSQL volume and checkpoint setup.

Docker and a live PostgreSQL/Slack workspace were not available in the current coding environment,
so those external-system checks remain explicit rather than being represented as passed.
