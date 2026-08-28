# Slack thread and Agent Session model

This note defines the runtime semantics for Slack threads, LangGraph conversations, Agent Sessions,
and individual question runs. It is an implementation contract for this repository, not a second
set of product requirements.

## Identity mapping

The application maps one Slack root thread to one durable agent conversation:

```text
conversation_id = <workspace team ID>:<channel ID>:<root thread timestamp>
LangGraph thread_id = conversation_id
Slack Agent Session = channel ID + root thread timestamp
```

Every incoming mention or follow-up candidate first has one durable `slack_turns` row keyed by the
Slack `event_id`. An accepted question also has an `agent_runs` row with status, model/tool
accounting, stream state, delivery state, and sources. The Slack parser derives UUIDv5 run/candidate
IDs from workspace, event ID, and purpose, so a Slack retry reproduces the same identity. Conversation,
turn, and run identities are intentionally different: one thread can contain many sequential turns,
but a retry of one Slack event cannot fork work.

## Message cases

| Slack message | Root timestamp used | Runtime result |
| --- | --- | --- |
| `@QA Agent ...` as a channel root message | The message's own `ts` | Starts a new Slack thread, conversation, LangGraph thread, and Agent Session. |
| `@QA Agent ...` inside an existing thread | The existing `thread_ts` | Creates a new question run in the same conversation and Agent Session. |
| Clear unmentioned question inside an agent-owned thread | The existing `thread_ts` | The responder judge authorizes a new run in the same conversation. |
| Ambiguous or human-directed reply inside an agent-owned thread | The existing `thread_ts` | The responder judge stays silent; no run is created. |
| Unmentioned channel root message | None | Ignored; it cannot silently establish bot ownership. |
| Mention in a different root message or channel | That message's root `ts` and channel | Starts a separate conversation with isolated checkpoints. |

Slack does not provide nested reply threads. A reply to a reply still carries the original root
`thread_ts`, so there is no application concept of a "subthread." Mentioning the bot again in an
existing thread therefore cannot start a nested session. It is an explicit new turn in the same
thread-scoped session.

## Who can continue a thread

The default policy is conversation-scoped, not requester-scoped. After an explicit mention creates
an agent-owned thread, any eligible non-guest human participant who can post in that channel may ask
a clear follow-up. This matches Slack's shared-thread model and the current authorization assumption
that the supplied SQLite file is one internal corpus available to eligible members of channels
containing the bot. Slack's
[Agent guidance](https://docs.slack.dev/ai/developing-agents/#members-only) says workspace guests
cannot access apps with the Agents feature enabled; the participant rule does not override that
platform restriction.

The bot does not infer that every reply is for it. Ordinary replies first pass these gates:

1. the message must be a human-authored channel thread reply with supported content;
2. the bot must already own the exact workspace/channel/root-thread identity;
3. a structured responder judge must return `respond` for a clear request to answer, explain,
   search, or verify;
4. `stay_silent`, `uncertain`, classifier failure, acknowledgements, and likely human-to-human
   discussion produce no bot run.

These semantic gates apply only to ordinary follow-up candidates. An explicit mention still enters
the same durable turn router and queue, but bypasses the responder classifier.

Setting `SLACK_ROUTING_POLICY=explicit_mentions_only` removes the semantic follow-up path. Explicit
mentions still use the same thread identity rule, so mentioning again in the thread remains a new
turn in the same conversation.

## Conversation history

LangGraph receives `conversation_id` as its `thread_id`. The checkpoint stores bounded semantic
history for that thread only. Before retrieval, a follow-up resolver may rewrite an elliptical
message such as "When does it renew?" into a standalone question using the recent bounded turns.
The resolver does not answer the question.

Every finalized outcome is a turn, including a greeting, clarification request, scope response, or
grounded answer. The history stores only the user message, final user-visible answer, and run ID;
it does not store private reasoning. A new root Slack thread gets a different checkpoint identity
even if its text and participants are identical.

This bound controls what is reused as model context; it is not a retention or erasure boundary.
PostgreSQL checkpoint revisions and the other durable Slack-derived records remain until the local
state is reset. The current take-home has no automated TTL or propagation from Slack message
deletion, user/workspace erasure, or app uninstallation. Production must minimize retained Slack
content and apply one verified deletion lifecycle across checkpoints, the application ledger,
Inngest state, traces, caches, and backups.

## Greeting, clarification, and scope behavior

- A whole-message greeting such as `hey` receives a short code-owned greeting without model or
  retrieval calls.
- A materially unclear message is classified at the existing retrieval-planning boundary and gets
  one concise clarification question. It does not search the corpus speculatively.
- If prior thread context resolves the ambiguity, the normal standalone-question resolution runs
  first and the message can proceed as a knowledge question.
- A clearly unrelated request receives a code-owned explanation of the bot's grounded knowledge-QA
  scope and an invitation to ask a relevant question.
- Only knowledge-question outcomes participate in evidence sufficiency, citation, and grounding
  checks. Direct conversational outcomes cannot be mislabeled as evidence failures.

When the final outcome asks for clarification, Slack delivery leaves the Agent Session in
`suspended`, meaning the agent is waiting for user input. A greeting, scope response, grounded
answer, failure, or cancellation leaves it `active` after terminal delivery.

## Multiple turns submitted quickly

Both app mentions and ordinary follow-up candidates enter the same `route-slack-turn` Inngest
function. PostgreSQL `slack_turns`, not Inngest concurrency, is the causal queue:

- every insert, claim, run link, and terminal transition takes a transaction-scoped advisory lock
  for the conversation;
- immutable event identity prevents one Slack event ID from being replayed with different message
  fields;
- pending rows are ordered by numeric Slack message timestamp, then creation time and event ID;
- a partial unique index permits only one `processing` row per conversation;
- later known turns retry through durable 15-second sleeps, bounded to 240 attempts (about one hour).

When an ordinary candidate becomes the head, the router verifies thread ownership and invokes the
responder classifier as a separately model-bounded child function. Explicit mentions skip that
child. A suppressed candidate reaches a terminal turn state without an agent run. For an accepted
turn, one PostgreSQL transaction creates and links its run; the router then invokes progress and the
primary question function in that order. It verifies the linked run is terminal before completing
the turn. This avoids a database-commit-then-new-Inngest-event dual-write window.

Ordering is exact for all turns already known when the head is claimed. A `processing` turn is not
preempted if a previously unseen event with an older Slack timestamp arrives later; the late event
waits behind the existing lease and is ordered with the remaining known rows afterward. This
intentional limitation favors one unambiguous owner over retroactive preemption.

Question, turn, and cancellation cleanup have separate responsibilities. If turn cleanup also
exhausts retries, the final dead-letter reconciler terminalizes or abandons a linked run before it
completes the turn. A poisoned `processing` row therefore cannot hold the conversation queue
forever.

## Stop targeting

Slack emits `agent_session_stopped` for a thread-scoped session, but application cancellation must
target an individual run. The handler therefore does not simply cancel the newest database row.
It validates the workspace/channel/root thread, compares reported streaming message timestamps
when present, uses event/message time as a causal upper bound when necessary, and atomically stores
the selected run and accepted/rejected result under the Stop event ID.

Stop and final delivery lock the same run:

- if cancellation commits before delivery is claimed, the result is withheld and the stream or
  session is finalized with `Stopped at your request.`;
- if final delivery has already been claimed, the late Stop is rejected and the verified answer
  completes;
- a retry of the same Slack Stop event returns its recorded outcome and cannot select a later turn.

Session status belongs to the run that successfully finalizes an outcome. Accepted cancellation
finalization leaves the session `active`; successful clarification delivery leaves it `suspended`.
A rejected or stale Stop, an event with no matching run, and cleanup that observes an
already-succeeded run do not write session status. This is intentional: delayed cleanup must not
reset a valid `suspended` clarification or overwrite a later run's session state.

Cancellation is cooperative. It is observed between durable steps, between graph progress stages,
and immediately before delivery. An already-running provider request may complete, but its result
is not delivered when cancellation won the transaction race.

## Supported Slack surfaces

The manifest and runtime currently support public and private channel threads. The bot must be a
channel member; use `/invite @QA Agent` and select the mention through Slack autocomplete.
Direct-message and multiparty-DM events are not
subscribed, so the Agent app container is not an alternate supported ingress in this take-home.
Adding it would require explicit `message.im`/`message.mpim` behavior, scopes, authorization rules,
and tests rather than silently treating it as equivalent to a shared channel.
