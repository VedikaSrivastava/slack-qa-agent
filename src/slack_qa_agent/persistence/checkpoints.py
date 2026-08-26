"""LangGraph-owned Postgres checkpoint schema initialization."""

from __future__ import annotations

import asyncio

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from slack_qa_agent.config import get_database_settings


async def setup_checkpoints() -> None:
    settings = get_database_settings()
    async with AsyncPostgresSaver.from_conn_string(settings.psycopg_database_url()) as saver:
        await saver.setup()


def main() -> None:
    asyncio.run(setup_checkpoints())


if __name__ == "__main__":
    main()
