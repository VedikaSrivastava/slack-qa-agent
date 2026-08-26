"""Persistence boundary for idempotent agent-run lifecycle updates."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.persistence.models import AgentRun, RunSource, RunStatus


class RunLedger(Protocol):
    async def create_queued(self, job: QuestionJob) -> tuple[uuid.UUID, bool]: ...

    async def attach_inngest_event(self, run_id: uuid.UUID, event_id: str) -> None: ...

    async def mark_running(self, run_id: uuid.UUID) -> None: ...

    async def get_completed_result(self, run_id: uuid.UUID) -> AgentResponse | None: ...

    async def mark_succeeded(self, run_id: uuid.UUID, response: AgentResponse) -> None: ...

    async def mark_failed(self, run_id: uuid.UUID, *, code: str, message: str) -> None: ...

    async def get_delivery(self, run_id: uuid.UUID) -> DeliveryState: ...

    async def set_placeholder(self, run_id: uuid.UUID, timestamp: str) -> None: ...

    async def set_response(self, run_id: uuid.UUID, timestamp: str) -> None: ...


@dataclass(frozen=True)
class DeliveryState:
    channel_id: str
    thread_ts: str
    placeholder_ts: str | None
    response_ts: str | None


class PostgresRunLedger:
    def __init__(
        self,
        engine: AsyncEngine,
        *,
        prompt_version: str,
        retrieval_version: str,
        model_name: str,
    ) -> None:
        self._engine = engine
        self._prompt_version = prompt_version
        self._retrieval_version = retrieval_version
        self._model_name = model_name

    async def create_queued(self, job: QuestionJob) -> tuple[uuid.UUID, bool]:
        run_id = job.agent_run_id
        statement = (
            pg_insert(AgentRun)
            .values(
                id=run_id,
                slack_event_id=job.event_id,
                conversation_id=job.conversation_id,
                status=RunStatus.QUEUED.value,
                slack_team_id=job.team_id,
                slack_channel_id=job.channel_id,
                slack_thread_ts=job.thread_ts,
                prompt_version=self._prompt_version,
                retrieval_version=self._retrieval_version,
                model_name=self._model_name,
            )
            .on_conflict_do_nothing(index_elements=[AgentRun.slack_event_id])
            .returning(AgentRun.id)
        )
        async with self._engine.begin() as connection:
            inserted = (await connection.execute(statement)).scalar_one_or_none()
            if inserted is not None:
                return inserted, True
            existing = (
                await connection.execute(
                    select(AgentRun.id).where(AgentRun.slack_event_id == job.event_id)
                )
            ).scalar_one()
            return existing, False

    async def attach_inngest_event(self, run_id: uuid.UUID, event_id: str) -> None:
        await self._update(run_id, inngest_event_id=event_id)

    async def mark_running(self, run_id: uuid.UUID) -> None:
        now = datetime.now(UTC)
        async with self._engine.begin() as connection:
            row = (
                await connection.execute(select(AgentRun.queued_at).where(AgentRun.id == run_id))
            ).one()
            queued_at = row[0]
            latency = max(0, int((now - queued_at).total_seconds() * 1000))
            await connection.execute(
                update(AgentRun)
                .where(
                    AgentRun.id == run_id,
                    AgentRun.status == RunStatus.QUEUED.value,
                )
                .values(
                    status=RunStatus.RUNNING.value,
                    started_at=now,
                    queue_latency_ms=latency,
                )
            )

    async def get_completed_result(self, run_id: uuid.UUID) -> AgentResponse | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(AgentRun.status, AgentRun.result_json).where(AgentRun.id == run_id)
                )
            ).one_or_none()
        if row is None or row.status != RunStatus.SUCCEEDED.value or row.result_json is None:
            return None
        return AgentResponse.model_validate(row.result_json)

    async def mark_succeeded(self, run_id: uuid.UUID, response: AgentResponse) -> None:
        now = datetime.now(UTC)
        async with self._engine.begin() as connection:
            started_at, queued_at = (
                await connection.execute(
                    select(AgentRun.started_at, AgentRun.queued_at).where(AgentRun.id == run_id)
                )
            ).one()
            started_at = started_at or queued_at
            await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    status=RunStatus.SUCCEEDED.value,
                    completed_at=now,
                    agent_latency_ms=max(0, int((now - started_at).total_seconds() * 1000)),
                    total_latency_ms=max(0, int((now - queued_at).total_seconds() * 1000)),
                    tool_call_count=response.tool_call_count,
                    retrieval_round_count=response.retrieval_round_count,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    insufficient_evidence=response.insufficient_evidence,
                    result_json=response.model_dump(mode="json"),
                    error_code=None,
                    sanitized_error_message=None,
                )
            )
            await connection.execute(delete(RunSource).where(RunSource.agent_run_id == run_id))
            if response.sources:
                await connection.execute(
                    insert(RunSource),
                    [
                        {
                            "id": uuid.uuid4(),
                            "agent_run_id": run_id,
                            "artifact_id": source.artifact_id,
                            "artifact_title": source.title,
                            "retrieval_rank": rank,
                            "retrieval_score": source.score,
                        }
                        for rank, source in enumerate(response.sources, start=1)
                    ],
                )

    async def mark_failed(self, run_id: uuid.UUID, *, code: str, message: str) -> None:
        await self._update(
            run_id,
            status=RunStatus.FAILED.value,
            completed_at=datetime.now(UTC),
            error_code=code[:128],
            sanitized_error_message=message[:2_000],
        )

    async def get_delivery(self, run_id: uuid.UUID) -> DeliveryState:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        AgentRun.slack_channel_id,
                        AgentRun.slack_thread_ts,
                        AgentRun.slack_placeholder_ts,
                        AgentRun.slack_response_ts,
                    ).where(AgentRun.id == run_id)
                )
            ).one()
        return DeliveryState(
            channel_id=row.slack_channel_id,
            thread_ts=row.slack_thread_ts,
            placeholder_ts=row.slack_placeholder_ts,
            response_ts=row.slack_response_ts,
        )

    async def set_placeholder(self, run_id: uuid.UUID, timestamp: str) -> None:
        await self._update(run_id, slack_placeholder_ts=timestamp)

    async def set_response(self, run_id: uuid.UUID, timestamp: str) -> None:
        await self._update(run_id, slack_response_ts=timestamp)

    async def _update(self, run_id: uuid.UUID, **values: Any) -> None:
        async with self._engine.begin() as connection:
            await connection.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))
