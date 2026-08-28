from __future__ import annotations

import asyncio
from types import TracebackType
from typing import cast

import pytest
from sqlalchemy.ext.asyncio import AsyncEngine

import knowledge_assistant.persistence.database as database_module
from knowledge_assistant.persistence.database import database_is_ready


class FailingConnectionContext:
    async def __aenter__(self) -> object:
        raise RuntimeError("database unavailable")

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


class FailingEngine:
    def connect(self) -> FailingConnectionContext:
        return FailingConnectionContext()


class SuccessfulConnection:
    def __init__(self) -> None:
        self.did_execute = False

    async def execute(self, statement: object) -> None:
        del statement
        self.did_execute = True


class SuccessfulConnectionContext:
    def __init__(self, connection: SuccessfulConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> SuccessfulConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


class SuccessfulEngine:
    def __init__(self) -> None:
        self.connection = SuccessfulConnection()

    def connect(self) -> SuccessfulConnectionContext:
        return SuccessfulConnectionContext(self.connection)


class WaitingConnection:
    async def execute(self, statement: object) -> None:
        del statement
        await asyncio.Event().wait()


class WaitingConnectionContext:
    async def __aenter__(self) -> WaitingConnection:
        return WaitingConnection()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


class WaitingEngine:
    def connect(self) -> WaitingConnectionContext:
        return WaitingConnectionContext()


async def test_readiness_translates_connection_failure_to_unavailable() -> None:
    engine = cast(AsyncEngine, FailingEngine())

    assert await database_is_ready(engine) is False


async def test_readiness_reports_available_after_successful_query() -> None:
    fake_engine = SuccessfulEngine()
    engine = cast(AsyncEngine, fake_engine)

    assert await database_is_ready(engine) is True
    assert fake_engine.connection.did_execute is True


async def test_readiness_connection_check_has_a_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A tiny timeout exercises the boundary without making the test depend on a real database.
    monkeypatch.setattr(database_module, "DATABASE_READINESS_TIMEOUT_SECONDS", 0.001)
    engine = cast(AsyncEngine, WaitingEngine())

    assert await database_is_ready(engine) is False
