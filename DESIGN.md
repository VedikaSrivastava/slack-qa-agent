# Design: Slack Based Q&A Chat Bot/Agent

I built a Slack QA agent over the supplied SQLite knowledge base.

The user experience is simple: mention the bot in a channel, ask a question, see real progress in slack, and receive answer in the same thread. Clear follow-ups can continue naturally without another mention. Use "Stop" to stop the bot.

What I cared about most was not making the agent look autonomous. I wanted to be able to explain what happens when retrieval is incomplete, slack retries an event, a model call takes too long or the user clicks Stop while work is still running. My goal was to make the system:

- grounded in the supplied knowledge base;
- bounded in retrieval, model calls, context, and repair;
- safe under slack retries and long-running model calls;
- durable across worker failures;
- explicit about what is guaranteed versus best-effort;
- measurable enough that I could reject regressions instead of optimizing by intuition.

The agent is stable at 6/7 strict official cases on each of three repeats, with no flaky strict cases, 21/21 operations-contract passes, ~10s median latency, and ~24s maximum latency in the accepted repeated run. The single persistent failure is a difficult corpus-wide comparative retrieval question (*which customer looks most likely to defect to a cheaper tactical competitor if we miss the next promised milestone, and what exactly is that milestone*). I also ran one LLM as a judge pass. Six of seven answers received 5/5 correctness, the judge found 0 material errors and 0 reference contradictions, and four answers passed the stricter quality gate that also requires 5/5 completeness.

### Flow

A normal interaction is:
1. A user mentions `@QA Agent` in an invited slack channel.
2. Slack shows agent progress tied to real workflow stages.
3. In the background, the agent resolves the question, retrieves evidence, grades coverage, drafts an answer, and verifies grounding.
4. Slack receives the complete verified answer in the same thread.
5. Non-mentioned follow ups:
   - A clear follow-up can continue without another mention.
   - The bot doesn't respond to human-to-human messages.
6. The user can use Slack’s native Stop control to stop the agent.

I do not stream draft answer tokens. The draft is verified after generation, so exposing it early would let users see claims that may later be repaired or removed. Also this is not a long running process, so just conveying that the agent is still running should be enough.

### What does success look like?
- Accurate answers grounded in the supplied knowledge base.
- Exact preservation of facts.
- Natural multi-turn slack behavior (without interrupting normal conversation).
- Fast slack acknowledgement with durable long-running work.
- Explicit ceilings for retrieval, model calls, tool calls, history, and repair.
- Safe partial answers, clarification, or abstention when evidence is insufficient.
- Separate measurement of correctness, completeness, retrieval, citations, latency, cost, and stability.

## High level system design
[View the diagram on Lucid](https://lucid.app/lucidchart/fbc24a17-ff0b-40ae-8db1-9f820f2e5d8a/edit?view_items=ksUob%2F765YdSspMlYZfBzN%2B9e3M%3D&page=0_0&invitationId=inv_07ae5edf-17f3-4799-86d9-fb5eac45cff3)
![Slack Bot System Design.png](./Slack%20Bot%20System%20Design.svg)

A slack event comes through a narrow /slack/events ingress, where FastAPI + Slack Bolt verify the request, and then the work is handed off to Inngest so I can acknowledge slack quickly instead of making slack wait on retrieval or model calls. Inngest owns durable execution and retries, while PostgreSQL owns the actual application state: turn ordering, runs, delivery/cancellation state, and langGraph checkpoints. This separation was important to me because LangGraph is good at expressing the reasoning workflow, but it should not also be responsible for slack retries, delivery races, or durable job execution. The supplied SQLite database stays separate and read-only as the knowledge source.   

I chose to use slack's *Events API* instead of *Socket Mode*. Socket mode would have made local setup easier, but I wanted one production-like, stateless ingress path instead of maintaining a separate WebSocket transport and app-level token just for development. That did create a setup blocker: slack needs a verified Request URL, but developer's tunnel URL does not exist when they first import the app. I solved that with a two-stage manifest: a bootstrap manifest for scopes/features, followed by a generated final manifest once the tunnel is running.  

The slack response surface also changed while I was building it. I initially used a placeholder message and *chat.update*, but that left the final answer looking like an edited message and I coudln't show progress properly. After rechecking slack's current agent API's, I switched to native agent sessions and task streaming. The agent now streams only code-owned progress steps while keeping the draft private until verification is complete. If native streaming fails, the verified final answer still has its own delivery path.

### Reliability, retries, and stop
I treat slack event_id as the idempotency key and derive the internal IDs from it. Direct mentions and follow-ups that might need an answer both go through the same Postgres-backed causal queue. Once a turn is accepted, the run gets created and linked to it in a single transaction, and only then does the durable router kick off progress and question processing. Stop was the tricky part. I write the cancellation down before handing it to Inngest, and stop and delivery fight over the same run: if stop gets there first, the answer is withheld while if delivery already claimed it a late stop doesn't do anything. Cancellation is cooperative, so a model call that's already in flight may run to completion and we don't deliver what it returns.

### Security
For the take-home, I kept the security boundary small and explicit. Slack Bolt verifies request signatures and timestamps, and the public tunnel exposes only POST /slack/events. Health, readiness, PostgreSQL, and the local Inngest endpoint are not public. I also run auth.test once at startup and verify the installed scopes there, so stale Slack authorization fails before the first user request instead of adding a network call to Slack's acknowledgement path. The knowledge database is opened read-only. The model never gets arbitrary SQL. It can only request the typed retrieval operations described below, with parameterized values and allowlisted identifiers.

### Observability
Logs, PostgreSQL run state, and Inngest execution timelines are the runtime source of truth, with optional local Langfuse tracing for model/graph debugging. I considered LangSmith first because of its LangChain integration, but hosted tracing would send prompts and outputs outside the local environment, while self-hosting it was more infrastructure than this take-home needed. Evaluation therefore writes versioned local reports.

## Langraph agent
[View the diagram on Lucid](https://lucid.app/lucidchart/f071b372-ccaa-47d3-9691-7fcf6a2667f4/edit?invitationId=inv_6775adf0-1e5e-44df-9af7-0135f2a49488)
![Agent Design](.//Slack%20agent%20.svg)

At a high level, the graph is pretty straightforward. The agent first figures out what the user is actually asking, using recent thread context when needed eg. follow ups. If it is a normal knowledge question, plan a small set of searches, retrieve evidence from the knowledge base, and check whether we have enough information to answer all parts of the question. If something important is missing, the agent gets one chance to search specifically for that gap. Once enough evidence is gathered, the agent generates the answer and verifies it against what was retrieved. Following this the agent can either return it, repair it once, or abstain if it still cannot support it. Greetings, unclear questions, and out-of-scope requests exit earlier instead of unnecessarily going through retrieval.

### Bounded graph
I wanted each question to have a predictable upper bound on work instead of letting an open-ended ReAct loop decide when it was done searching otherwise it becomes hard to reason about latency, cost, and failure behavior, and one long-running question can keep consuming shared model capacity while other questions wait. So the agent is capped at 2 retrieval rounds, 8 retrieval/tool calls, 9 model calls, and 1 answer repair. In the final evaluation, runs averaged 1.19 retrieval rounds, 4.24 tool calls, and 4.38 model calls, with no budget violations, so these are safety ceilings rather than normal usage. If the agent reaches a ceiling, it answers the parts it can support and explicitly leaves the remaining gap unanswered; if there is not enough evidence to give a useful grounded answer, it abstains.

### Retrival
The model never generates arbitrary SQL. The planner produces bounded retrieval intents, and code maps those intents to a small allowlisted set of operations (FTS5 search, structured account lookup, and artifact reads). The SQLite connection is read-only, values are parameterized, and identifiers are allowlisted. I chose lexical + structured retrieval because the dataset has a lot of exact operational information like customer names, commands, schema/config names, etc. Structured filtering is also better for questions that ask for an exact cohort of accounts rather than semantically similar ones. 

One issue I ran into was with broad comparison questions. A global BM25 ranking could over-represent one heavily documented customer, and scores from two differently worded searches are not really comparable to each other. I added a scenario-first pass and, for multi-query comparisons, preserved the best unique evidence from each planned search before filling the remaining slots. That prevents one customer or one search dimension from immediately consuming the evidence budget without increasing the overall tool-call ceiling.

The limitation is that lexical retrieval is weaker when the question expresses the right concept using very different wording from the corpus. The BlueHarbor comparative case is the clearest example: its lexical coverage was only 0.0769, far below the other cases, and it remained the only strict failure. I did not add a vector database just to patch one benchmark case; my next step would be to build a broader comparative robustness suite and then test hybrid semantic retrieval or reranking with the model and action budgets held fixed.

### Grounding and repair
After the initial answer is generated, the verifier checks it against the retrieved evidence for unsupported factual claims, wrong exact values, omitted requested facts, and invalid citations. If one of those grounding problems is found the agent gets one repair attempt and one re-check. If the answer still cannot be supported, I would rather return only the supported part or abstain than send something plausible-looking to Slack.

One thing I learned while building this was that not every quality issue belongs in the verifier. I initially made it reject answers for presentation problems as well, and a previously correct case started intermittently abstaining because a style problem now had the authority to kill the entire answer. I moved exact formatting leaks to deterministic sanitization, kept wording preferences in generation, and reserved the verifier for issues serious enough to justify rejecting an answer.

### Conversation context
Each Slack root thread maps to one LangGraph conversation, but I keep only the most recent 6 finalized turns as semantic history. For each turn I also save compact provenance: the clean answer, cited source references, and the IDs of the retrieved artifacts rather than copying the full evidence into the checkpoint. That lets the agent resolve follow-ups such as “what about that one?” using recent context without letting the prompt grow with the entire thread. Hence if a follow-up depends on evidence from an earlier answer, the agent can re-read those artifact IDs from the immutable SQLite database instead of treating the model's previous answer as evidence. That prevents an error in one turn from becoming the “ground truth” for the next turn. Whether an unmentioned slack reply is actually meant for the bot is handled before this graph by the conservative thread router; once a message is accepted as a question, the graph only has to focus on resolving and answering it.

## Evaluations

I used a lexicographic objective rather than one aggregate score:
1. factual correctness, exact values, grounding, answerability, and safety;
2. completeness of requested parts, customer sets, and groupings;
3. bounded and explainable agent actions;
4. latency, cost, and answer length as tie-breakers.

That ordering mattered because several experiments improved one surface while making another materially worse. *For example*, increasing the retrieval budget made the difficult case slower without fixing it and regressed an already-correct cohort question. A presentation rule initially made answers look cleaner but accidentally caused valid answers to abstain.

### How I ran experiments

I tried to change one major variable at a time so I could understand what actually caused a result. The seven official questions stayed unchanged throughout development. My own derived, multi-turn, routing, and robustness cases stayed separate so I was not adding runtime behavior around known benchmark answers. 
I also learned not to trust a single model run. One prompt change initially scored 6/7 and looked fine. When I repeated the exact same configuration three times, it scored 6/7, 6/7, 5/7 and exposed a new flaky failure. After that my rule became:
- one run can reject a bad idea quickly;
- one run is not enough to accept a change;
- anything I want to ship runs the official suite at least three times;
- there should be no flaky strict cases;
- if something fails, I inspect that case before changing another variable.

Three repeats are still a small sample, so I treat this as a stability check rather than a production accuracy estimate. I kept deterministic and semantic evaluation separate because they answer different questions. The deterministic evaluator checks things I can measure exactly:
- required commands and dates;
- required customers for exhaustive-set questions;
- answer vs. insufficient-evidence behavior;
- citation and source integrity;
- retrieval, model, and tool-call budgets;
- safety checks where applicable.

The LLM judge is for properties that are harder to capture with exact matching, mainly semantic correctness, completeness, conciseness, material errors, and agreement with the reference answer.

*Metric definitions*
I report a few deterministic sub-metrics separately because they diagnose different failure modes:
- Exact-content checks: measure requirements like required commands, dates, customers, or grouped outputs.
- Evidence contract: checks whether the answer stays within the retrieved evidence.
- Retrieval diagnostic measures whether retrieval surfaced artifacts that support the reference answer for diagnosis rather than correctness because another artifact can support the same answer.
- Citation diagnostic: measures whether the cited artifacts support the reference answer. It has the same limitation, another artifact can support the same answer.
- Citation integrity: every emitted artifact citation must resolve to evidence actually available to the agent.

I kept the operations contract separate as well which evaluate runtime behavior and execution rather than whether the final answer is semantically correct. 

The seven official questions and reference answers are 'src/knowledge_assistant/evals/cases/full.json'. '--suite' full loads that file. Case ids like official-blueharbor-defection-risk, are the id fields there. 

### Main experiments

Glossary:
- p*xx* : prompt version
- r*xx*: retrival version

| Experiment | What I changed | What happened | Decision |
|---|---|---|---|
| Historical prompt/retrieval iterations | `p15/r8` → `p19/r12` | 6/7 → 5/7 → 6/7 → 6/7 → 6/7 | `p16/r9` regressed and was rejected |
| Model routing | Mostly single-model `gpt-4.1-mini` → role-specific GPT-5.4 / Mini / Nano | Both reached 6/7, but the split let me reserve the stronger model for planning/synthesis and use smaller models for structured tasks | Kept role-specific routing |
| Retrieval diversification | Global BM25 → scenario-first candidate selection + query-diverse merging | Prevented one heavily documented customer or one planned search from immediately dominating the evidence budget | Kept bounded diversification |
| Prompt v20 | Exact-command formatting, milestone-date chains, comparison disambiguation | Stayed at 6/7 | Kept |
| Prompt v21 | Better `continue / retry / resume` behavior from thread context | Stayed at 6/7; fixed a Slack behavior outside the official suite | Kept |
| First prompt v22 attempt | Added presentation rules and let the verifier reject retrieval-internal wording | First run 6/7; three repeats 6/7, 6/7, 5/7; `official-canada-approval-pattern` became flaky | Rejected verifier rule |
| Shipped prompt v22 (details in next section) | Kept presentation guidance, moved exact leaks to deterministic sanitization, removed style enforcement from verifier | 6/7, 6/7, 6/7, no flaky strict cases | Shipped |
| Larger retrieval budget | Retrieval rounds 2 → 3, tool calls 8 → 10, everything else fixed | `official-blueharbor-defection-risk` and `official-canada-approval-pattern` both used the third round and still failed; targeted slice of 3 hard official cases scored 1/3; the two failing cases used the third round and still failed | Rejected; full suite not run |
| Semantic judge | One LLM judge pass after the deterministic profile was stable | Found two completeness gaps the deterministic contract missed | Kept as a separate diagnostic |

FYI, the historical `p15/r8`–`p19/r12` experiments are not the final configuration. The main thing I tried to avoid was changing several knobs together and then not knowing which one helped. If a more sophisticated-looking change regressed a question that already worked, I reverted it. The larger-budget experiment was also useful because it narrowed the remaining problem. The difficult BlueHarbor comparison used the extra retrieval round and still failed, so I treat the remaining issue as a retrieval-quality / evidence-selection problem rather than simply a retrieval-budget problem.

### Final Agent evaluation

- Profile: split-gpt-5.4-hybrid
- Prompt: v22
- Retrieval: v12
- Evaluation protocol: v13
- Repeats: 3
- Official questions: 7
- Total case-runs: 21

| Measure | Result |
|---|---:|
| Strict contract per repeat | 6/7 (all 3 runs) |
| Pooled strict result | 18/21 |
| Flaky strict cases | 0 |
| Exact-content checks | 75% of applicable checks|
| Evidence contract | ~90%|
| Operations contract | 100% (21/21)|
| Required customer recall | 100% on applicable cases|
| Citation integrity | 100%|
| Retrieval diagnostic | ~86%|
| Citation diagnostic | ~81%|
| Retrieval rounds | ~1 mean / 2 max|
| Tool calls | 89 total / ~4 per case-run|
| Model calls | 92 total / ~4 per case-run|
| Tokens| 544,947 total / ~25,950 per case-run|
| Latency | ~10s p50 / ~21s p95 / ~24s max|
| Budget violations | 0|

### Final LLM as a Judge pass

The judge's quality gate requires:
- correctness >= 4/5;
- completeness = 5/5;
- no material error;
- no contradiction with the reference.

I made completeness the stricter gate because an answer that omits a requested sub-part is still incomplete even if every fact it does include is correct.

| Measure | Result |
| --- | ---: |
| Strict contract | 6/7|
| Judge answer-quality gate |4/7|
| Combined task-quality gate |4/7 |
| Material errors |0 |
| Reference contradictions |0|
| Mean correctness |4.43/5 |
| Mean completeness | 4.14/5|
| Mean conciseness | 3.57/5|

Per case results:
| Case | Correctness | Completeness | Conciseness | Result |
|---|---:|---:|---:|---|
| BlueHarbor proof-plan | 5 | 5 | 2 | Pass; complete but verbose |
| Verdant rollback | 5 | 5 | 3 | Pass |
| MapleHarvest mappings | 5 | 4 | 4 | Correct, but missed “define alias mappings” |
| Aureum SCIM | 5 | 4 | 4 | Correct, but missed SCIM tracing / approval-latency detail |
| BlueHarbor defection-risk | 1 | 1 | 5 | Abstained; did not answer the requested comparison |
| NA West account-groups | 5 | 5 | 3 | Pass |
| Canada approval-pattern | 5 | 5 | 4 | Pass |

Two answers were correct but incomplete by one requested detail:
- MapleHarvest: missed “define alias mappings” as a workshop outcome.
- Aureum: missed Jin's SCIM-tracing / approval-latency proposal.

Those misses are also why I keep retrieval and final-answer quality separate. The BlueHarbor comparison remains the real semantic failure. In this judge run the agent abstained instead of fabricating the milestone or contradicting the reference. That is the safer behavior, but I still count it as a failed answer.

#### Observations

- One run can be misleading. The verifier regression looked fine until I repeated it.
- More agent work is not automatically better. The third retrieval round added latency without fixing the hard cases.
- Retrieval budget and retrieval quality are different problems. Increasing the budget did not fix the remaining comparison failure.
- Retrieval success and answer completeness are different failure modes. Evidence can be available and still be omitted during synthesis.
- Different evaluation layers catch different failures. Exact checks caught dates, sets, citations, and budgets; the semantic judge caught completeness gaps.
- Guardrails need the right authority. A presentation issue should not have the same consequence as an unsupported factual claim, giving the verifier that authority caused otherwise valid answers to be rejected.
- An abstention is still a failure. I prefer it to hallucination, but I do not count it as success.
- Agent quality is not only answer quality. I evaluate bounded execution and runtime behavior separately from semantic correctness.
- I did not optimize the runtime to force 7/7. The remaining failure points to a real comparative-retrieval weakness that I would rather test on broader held-out cases before adding semantic/vector retrieval.

## Gaps and known issues

The core retrieval and delivery path is stable, but there are still a few places where the live Slack experience is weaker than the evaluation numbers alone suggest. I would rather call those out directly than hide them behind extra agent loops or benchmark-specific fixes.

- *Follow-up resolution is still the weakest conversational behavior.* 
The agent keeps bounded conversation state and can resolve normal follow-ups such as a customer name or “what about the rollback?”. It is less reliable when the follow-up only supplies scope or intent. For example, after asking which company's products to list, the reply `every` led to another clarification instead of being resolved to “list products for every company.” The same issue can appear with replies such as `both`, `all of them`, or `the other one`. I would add held-out cases for these patterns and give the resolver explicit information about whether the previous agent turn was a clarification before considering a stronger model.

- *Unmentioned follow-up routing intentionally favors silence.* 
Explicit `@QA Agent` mentions always enter the graph where as normal replies in an agent-owned thread first go through a lightweight classifier to decide whether the message is for the agent or part of the human conversation. I bias ambiguous cases toward staying silent because interrupting people in a shared Slack thread is worse than occasionally requiring another mention. The downside is that vauge or small but legitimate follow-ups can be missed. I would evaluate false interruptions and missed follow-ups separately before making this classifier more aggressive.

- *Some conversational intent can still be misclassified.* 
Greetings, capability questions, out-of-scope messages, and clarification requests can exit before retrieval and this saves unnecessary model/retrieval work, but a wrong classification becomes a complete miss. In the live thread, `can you check the conversation history` was interpreted like a capability question and returned the generic capability response. I would add explicit handling and eval cases for thread-history rather than letting them fall into the general capability bucket.

## Production scalability

### What already scales reasonably well

- Slack ingress is lightweight and stateless: request path verifies the Slack event, hands durable work to inngest, and acknowledges slack instead of running model work synchronously.
- Execution scales across conversations: inngest handles retries, cancellation, and concurrency, while PostgreSQL stores runs, conversation state, stop state, and delivery state. I intentionally serialize work within a single Slack thread because later turns can depend on earlier ones.
- Agent work is bounded
- Model routing separates cost from reasoning quality: smaller models handle structured routing and resolution work while the stronger model is reserved for planning and synthesis. Although I would want to experiment with other models and providers, i believe it could improve agent performance.

At larger scale, I expect model concurrency, latency, and token spend to become the bottleneck before FastAPI or Slack ingress. I would use Inngest concurrency as backpressure and monitor queue depth, OpenAI rate limits, token spend, p95 latency, abstention rate, and routing suppress rate before simply raising concurrency. I would not add generated-answer caching by default. A safe cache key would need to include authorization scope, TTL, and invalidation rules otherwise the same question could return stale or unauthorized evidenc

### What I would change for production

- Replace SQLite as the knowledge plane
SQLite is a good fit for a fixed, read-only take-home corpus, but every application replica currently assumes the same immutable file. For a changing or much larger corpus I would keep the existing typed retrieval interface and move the underlying store to PostgreSQL FTS or a search service. I would also version the corpus so every run records which knowledge snapshot it queried. Hybrid/vector retrieval would be an evaluated improvement, not the default architecture.

- Add workspace and user-level authorization at retrieval time.
Today the database behaves like one shared internal corpus. In production every run would carry a workspace and requester identity, and every artifact would have an ACL or permission scope. Search, lookup, cohort enumeration, and re-reading artifacts from conversation history would all enforce that permission context before evidence reaches the model. I would not ask the LLM to decide whether somebody is allowed to see a document.

- Treat the Slack channel as an authorization boundary.
Document ACLs are not enough if the bot posts a sensitive answer into a channel where other members can read it. We would need channel allowlists, membership checks, or private/DM delivery. Authorization has to answer both “can this requester retrieve this?” and “who will be able to read the resulting Slack message?”

I would not introduce unlimited retrieval loops, extra critic agents, answer caching, or a vector database without evidence that they solve a measured problem. Caching in particular becomes an authorization problem once different users can ask the same question with different access.

If I were taking the system beyond the assignment, my order would be:

1. Expand the routing and multi-turn evals around the live failures.
2. Add Slack OAuth and workspace isolation.
3. Add artifact-level ACLs and channel-level disclosure policy.
4. Replace the immutable SQLite corpus with a versioned production knowledge plane.
5. Add retention, spend, queue-depth, and provider-rate-limit monitoring.
6. Test hybrid retrieval on held-out comparative questions.