"""Persistence boundary for idempotent agent-run lifecycle updates."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

from sqlalchemy import delete, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.persistence.models import AgentRun, RunSource, RunStatus


class RunTransitionError(RuntimeError):
    """Raised when a persisted run receives an illegal lifecycle transition."""


_ALLOWED_RUN_TRANSITIONS = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.FAILED}),
    RunStatus.RUNNING: frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED}),
}
_IDEMPOTENT_TERMINAL_STATUSES = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)


def should_apply_run_transition(current: RunStatus, target: RunStatus) -> bool:
    """Validate a lifecycle transition and identify idempotent retry replays."""

    # The running replay covers a committed mark-started step whose acknowledgement was lost.
    if current == target == RunStatus.RUNNING:
        return False
    if current == target and current in _IDEMPOTENT_TERMINAL_STATUSES:
        return False
    if target in _ALLOWED_RUN_TRANSITIONS.get(current, frozenset()):
        return True
    raise RunTransitionError(f"Illegal agent-run transition: {current.value} -> {target.value}")


class RunLedger(Protocol):
    async def create_queued(self, job: QuestionJob) -> tuple[uuid.UUID, bool]: ...

    async def attach_inngest_event(self, run_id: uuid.UUID, event_id: str) -> None: ...

    async def mark_running(self, run_id: uuid.UUID) -> None: ...

    async def get_persisted_agent_result(self, run_id: uuid.UUID) -> AgentResponse | None: ...

    async def persist_agent_result(self, run_id: uuid.UUID, response: AgentResponse) -> None: ...

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


@dataclass(frozen=True)
class _LockedRunState:
    status: RunStatus
    queued_at: datetime
    started_at: datetime | None
    result_json: dict[str, Any] | None


async def _lock_run_for_transition(
    connection: AsyncConnection, run_id: uuid.UUID
) -> _LockedRunState:
    """Lock one run so status validation and mutation share one transaction."""

    row = (
        await connection.execute(
            select(
                AgentRun.status,
                AgentRun.queued_at,
                AgentRun.started_at,
                AgentRun.result_json,
            )
            .where(AgentRun.id == run_id)
            .with_for_update()
        )
    ).one_or_none()
    if row is None:
        raise RunTransitionError(f"Agent run does not exist: {run_id}")
    return _LockedRunState(
        status=RunStatus(row.status),
        queued_at=row.queued_at,
        started_at=row.started_at,
        # SQLAlchemy's JSON boundary is dynamically typed; Pydantic validates it before use.
        result_json=row.result_json,
    )


def _require_single_run_update(updated_row_count: int, run_id: uuid.UUID) -> None:
    if updated_row_count != 1:
        raise RunTransitionError(
            f"Expected to update one agent run {run_id}, updated {updated_row_count}"
        )


def _responses_match(persisted_json: dict[str, Any], response: AgentResponse) -> bool:
    return AgentResponse.model_validate(persisted_json) == response


def _should_persist_agent_result(
    run: _LockedRunState,
    run_id: uuid.UUID,
    response: AgentResponse,
) -> bool:
    """Accept an idempotent replay but reject a second, conflicting agent result."""

    if run.status != RunStatus.RUNNING:
        raise RunTransitionError(
            f"Cannot persist an agent result while run {run_id} is {run.status.value}"
        )
    if run.result_json is not None:
        if not _responses_match(run.result_json, response):
            raise RunTransitionError(f"Agent run already has a different result: {run_id}")
        return False
    return True


def _require_matching_persisted_result(
    run: _LockedRunState,
    run_id: uuid.UUID,
    response: AgentResponse,
) -> None:
    if run.result_json is None:
        raise RunTransitionError(f"Agent run has no persisted result: {run_id}")
    if not _responses_match(run.result_json, response):
        raise RunTransitionError(f"Agent run result changed before completion: {run_id}")


def _require_agent_result_readable(status: RunStatus, run_id: uuid.UUID) -> None:
    if status not in {RunStatus.RUNNING, RunStatus.SUCCEEDED}:
        raise RunTransitionError(
            f"Cannot reuse an agent result while run {run_id} is {status.value}"
        )


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
        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            if not should_apply_run_transition(run.status, RunStatus.RUNNING):
                return
            now = datetime.now(UTC)
            latency = max(0, int((now - run.queued_at).total_seconds() * 1000))
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    status=RunStatus.RUNNING.value,
                    started_at=now,
                    queue_latency_ms=latency,
                )
            )
            _require_single_run_update(result.rowcount, run_id)

    async def get_persisted_agent_result(self, run_id: uuid.UUID) -> AgentResponse | None:
        """Return agent output persisted before Slack delivery, if one exists."""

        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(AgentRun.status, AgentRun.result_json).where(AgentRun.id == run_id)
                )
            ).one_or_none()
        if row is None:
            raise RunTransitionError(f"Agent run does not exist: {run_id}")
        _require_agent_result_readable(RunStatus(row.status), run_id)
        if row.result_json is None:
            return None
        return AgentResponse.model_validate(row.result_json)

    async def persist_agent_result(self, run_id: uuid.UUID, response: AgentResponse) -> None:
        """Persist model output while the run remains active for retry-safe reuse."""

        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            if not _should_persist_agent_result(run, run_id, response):
                return
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(result_json=response.model_dump(mode="json"))
            )
            _require_single_run_update(result.rowcount, run_id)

    async def mark_succeeded(self, run_id: uuid.UUID, response: AgentResponse) -> None:
        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            should_update = should_apply_run_transition(run.status, RunStatus.SUCCEEDED)
            _require_matching_persisted_result(run, run_id, response)
            if not should_update:
                return
            if run.started_at is None:
                raise RunTransitionError(f"Running agent run has no started_at timestamp: {run_id}")
            now = datetime.now(UTC)
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    status=RunStatus.SUCCEEDED.value,
                    completed_at=now,
                    agent_latency_ms=max(0, int((now - run.started_at).total_seconds() * 1000)),
                    total_latency_ms=max(0, int((now - run.queued_at).total_seconds() * 1000)),
                    tool_call_count=response.tool_call_count,
                    model_call_count=response.model_call_count,
                    retrieval_round_count=response.retrieval_round_count,
                    input_tokens=response.input_tokens,
                    output_tokens=response.output_tokens,
                    insufficient_evidence=response.insufficient_evidence,
                    error_code=None,
                    sanitized_error_message=None,
                )
            )
            _require_single_run_update(result.rowcount, run_id)
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
        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            if not should_apply_run_transition(run.status, RunStatus.FAILED):
                return
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    status=RunStatus.FAILED.value,
                    completed_at=datetime.now(UTC),
                    error_code=code[:128],
                    sanitized_error_message=message[:2_000],
                )
            )
            _require_single_run_update(result.rowcount, run_id)

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
            result = await connection.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(**values)
            )
            _require_single_run_update(result.rowcount, run_id)
