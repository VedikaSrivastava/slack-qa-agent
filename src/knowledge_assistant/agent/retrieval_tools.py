"""Async retrieval tools exposed to the bounded answer workflow."""

from __future__ import annotations

import asyncio
from time import perf_counter

import structlog

from knowledge_assistant.retrieval.models import (
    AccountLookupInput,
    EvidenceItem,
    ReadArtifactsInput,
    SearchHit,
    SearchKnowledgeInput,
)
from knowledge_assistant.retrieval.repository import SQLiteKnowledgeRepository

logger = structlog.get_logger(__name__)


class KnowledgeRetrievalTools:
    """Async, observable adapter over the synchronous SQLite repository."""

    def __init__(self, repository: SQLiteKnowledgeRepository) -> None:
        self._repository = repository

    async def search_knowledge(self, request: SearchKnowledgeInput) -> list[SearchHit]:
        started = perf_counter()
        results = await asyncio.to_thread(self._repository.search, request)
        logger.info(
            "knowledge_search_completed",
            duration_ms=round((perf_counter() - started) * 1_000),
            result_count=len(results),
            limit=request.limit,
        )
        return results

    async def read_artifacts(self, request: ReadArtifactsInput) -> list[EvidenceItem]:
        started = perf_counter()
        results = await asyncio.to_thread(self._repository.read, request)
        logger.info(
            "knowledge_read_completed",
            duration_ms=round((perf_counter() - started) * 1_000),
            artifact_count=len(results),
            context_chars=sum(len(item.content) for item in results),
        )
        return results

    async def lookup_accounts(self, request: AccountLookupInput) -> list[EvidenceItem]:
        started = perf_counter()
        results = await asyncio.to_thread(self._repository.lookup_accounts, request)
        logger.info(
            "account_lookup_completed",
            duration_ms=round((perf_counter() - started) * 1_000),
            account_count=len(results),
            region=request.region,
            country=request.country,
            product=request.product,
        )
        return results
