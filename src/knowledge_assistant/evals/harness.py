"""Shared setup for the Postgres-free offline evaluation path.

Production builds the processor with a Postgres checkpointer so a durable run can resume
across restarts. Evaluation does not need that: every case runs to completion in one process.
These helpers give the graph an in-process :class:`InMemorySaver` and a placeholder
``DATABASE_URL`` (never connected to) so ``python -m knowledge_assistant.evals`` runs with
nothing but an ``OPENAI_API_KEY`` and the bundled SQLite file.
"""

from __future__ import annotations

import os
from pathlib import Path

from langgraph.checkpoint.memory import InMemorySaver

from knowledge_assistant.config import AgentRuntimeSettings

# Satisfies ``DatabaseSettings`` URL validation. The offline harness always passes an
# ``InMemorySaver`` into ``create_question_processor``, so this string is never dialed.
_PLACEHOLDER_DATABASE_URL = "postgresql+asyncpg://eval:eval@127.0.0.1:5432/eval-unused"

# Production sets provider retries to 0 because Inngest owns durable retry. Offline evaluation
# has no such layer, so a transient TLS/connection blip would otherwise fail a whole repeat
# (7+ model calls of work). A few SDK-level retries with backoff absorb those.
EVAL_MAX_RETRIES = 4


def load_eval_agent_settings(env_file: Path | None) -> AgentRuntimeSettings:
    """Load agent settings for an offline run, defaulting the unused ``DATABASE_URL``."""

    if env_file is not None and not env_file.is_file():
        raise FileNotFoundError(f"Environment file does not exist: {env_file}")
    os.environ.setdefault("DATABASE_URL", _PLACEHOLDER_DATABASE_URL)
    os.environ.setdefault("APP_ENV", "test")
    return AgentRuntimeSettings(_env_file=env_file)


def new_eval_checkpointer() -> InMemorySaver:
    """One fresh in-process checkpointer per evaluation process."""

    return InMemorySaver()
