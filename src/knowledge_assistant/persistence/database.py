"""Async SQLAlchemy engine construction and lightweight readiness checks."""

from __future__ import annotations

import asyncio

import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from knowledge_assistant.config import DatabaseSettings

DATABASE_READINESS_TIMEOUT_SECONDS = 2.0

logger = structlog.get_logger(__name__)


def create_database_engine(settings: DatabaseSettings) -> AsyncEngine:
    return create_async_engine(
        settings.sqlalchemy_database_url(),
        pool_pre_ping=True,
        pool_recycle=300,
    )


async def database_is_ready(engine: AsyncEngine) -> bool:
    try:
        async with asyncio.timeout(DATABASE_READINESS_TIMEOUT_SECONDS):
            async with engine.connect() as connection:
                await connection.execute(text("SELECT 1"))
    except Exception as exc:
        # Readiness is an isolation boundary: report unavailable without taking down liveness.
        logger.warning(
            "database_readiness_failed",
            error_code="database_readiness_failed",
            exception_class=type(exc).__name__,
        )
        return False
    return True
