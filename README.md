# Slack Q&A Agent

A production-minded Slack question-answering agent grounded in the provided SQLite knowledge base.

## Architecture boundary

The project deliberately separates three concerns:

1. **Slack transport** validates and normalizes inbound events.
2. **Inngest execution** queues and durably executes long-running work.
3. **LangGraph agent logic** handles conversation state, retrieval, evidence grading, and answer generation.

Slack and Inngest are adapters around a testable agent service. The same agent can be invoked from Slack, a CLI, or the evaluation runner without changing its reasoning logic.

```text
Slack Events API
      |
      v
FastAPI + Slack Bolt  -- enqueue --> Inngest
                                      |
                                      v
                               Question Processor
                                /              \
                               v                v
                         LangGraph Agent   Slack Publisher
                               |
                               v
                         Read-only SQLite
```

## Why two orchestration layers?

- **LangGraph** owns agent orchestration: state, tool use, bounded retrieval loops, and multi-turn memory.
- **Inngest** owns execution orchestration: fast webhook handoff, queueing, retries, idempotency, concurrency, and run observability.

The first implementation will treat the complete LangGraph invocation as a coarse-grained durable job. We will not wrap every graph node in an Inngest step because duplicate orchestration and checkpointing would make failure semantics harder to reason about.

## Local setup

Prerequisites: Python 3.12, `uv`, and the Inngest CLI.

```bash
cp .env.example .env
uv sync --all-groups
uv run uvicorn slack_qa_agent.main:app --reload --port 8000
```

In another terminal:

```bash
inngest dev -u http://localhost:8000/api/inngest
```

Health checks:

```bash
curl http://localhost:8000/healthz
curl http://localhost:8000/readyz
```

## Current status

This branch establishes project structure, dependency injection boundaries, configuration, logging, Slack event normalization, and an Inngest dispatcher. Retrieval and the final LangGraph workflow are intentionally the next implementation phase.

> `DESIGN.md` is intentionally absent from this scaffold. The assignment requires it to be written entirely by the candidate.
