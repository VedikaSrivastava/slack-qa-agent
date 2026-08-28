"""LangGraph-owned Postgres checkpoint schema initialization."""

from __future__ import annotations

import asyncio
import selectors
import sys

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from knowledge_assistant.config import get_database_settings


async def setup_checkpoints() -> None:
    settings = get_database_settings()
    async with AsyncPostgresSaver.from_conn_string(settings.psycopg_database_url()) as saver:
        await saver.setup()


def _checkpoint_loop_factory() -> asyncio.AbstractEventLoop:
    return _build_checkpoint_loop(sys.platform)


def _build_checkpoint_loop(platform_name: str) -> asyncio.AbstractEventLoop:
    if platform_name == "win32":
        # psycopg async connections require a selector loop on Windows; the
        # asyncio default is a ProactorEventLoop.
        return asyncio.SelectorEventLoop(selectors.SelectSelector())
    return asyncio.new_event_loop()


def main() -> None:
    with asyncio.Runner(loop_factory=_checkpoint_loop_factory) as runner:
        runner.run(setup_checkpoints())


if __name__ == "__main__":
    main()
