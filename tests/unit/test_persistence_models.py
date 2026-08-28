from __future__ import annotations

import uuid
from types import TracebackType
from typing import Any, cast

from sqlalchemy import Table
from sqlalchemy.ext.asyncio import AsyncEngine

from knowledge_assistant.agent.models import AgentResponse, QuestionDisposition
from knowledge_assistant.persistence.models import (
    AgentRun,
    DeliveryStatus,
    RunDeliveryPart,
    RunStatus,
    SlackStopEvent,
    SlackStoppedStream,
    SlackStreamMode,
    SlackStreamState,
    SlackTurn,
)
from knowledge_assistant.persistence.repositories import CancellationClaim, PostgresRunLedger


class _ScalarResult:
    def __init__(self, value: Any) -> None:
        self._value = value

    def scalar_one_or_none(self) -> Any:
        return self._value


class _RecordingConnection:
    def __init__(self, value: Any) -> None:
        self._value = value
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _ScalarResult:
        self.statements.append(statement)
        return _ScalarResult(self._value)


class _ConnectionContext:
    def __init__(self, connection: _RecordingConnection) -> None:
        self._connection = connection

    async def __aenter__(self) -> _RecordingConnection:
        return self._connection

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, exc_value, traceback


class _RecordingEngine:
    def __init__(self, value: Any) -> None:
        self.connection = _RecordingConnection(value)

    def connect(self) -> _ConnectionContext:
        return _ConnectionContext(self.connection)


class _ReplayRow:
    def __init__(self, run_id: uuid.UUID | None, accepted: bool) -> None:
        self.agent_run_id = run_id
        self.accepted = accepted


class _ReplayResult:
    def __init__(
        self,
        *,
        scalar: str | None = None,
        row: _ReplayRow | None = None,
    ) -> None:
        self._scalar = scalar
        self._row = row

    def scalar_one_or_none(self) -> str | None:
        return self._scalar

    def one(self) -> _ReplayRow:
        assert self._row is not None
        return self._row


class _ReplayConnection:
    def __init__(self, run_id: uuid.UUID, accepted: bool) -> None:
        self._results = [
            _ReplayResult(scalar=None),
            _ReplayResult(row=_ReplayRow(run_id, accepted)),
        ]
        self.statements: list[Any] = []

    async def execute(self, statement: Any) -> _ReplayResult:
        self.statements.append(statement)
        return self._results.pop(0)


class _ReplayEngine:
    def __init__(self, run_id: uuid.UUID, accepted: bool) -> None:
        self.connection = _ReplayConnection(run_id, accepted)

    def begin(self) -> _ConnectionContext:
        return _ConnectionContext(cast(_RecordingConnection, self.connection))


class _ProgressHeadConnection(_RecordingConnection):
    def __init__(self, conversation_id: str, head_run_id: uuid.UUID) -> None:
        super().__init__(None)
        self._values: list[str | uuid.UUID] = [conversation_id, head_run_id]

    async def execute(self, statement: Any) -> _ScalarResult:
        self.statements.append(statement)
        return _ScalarResult(cast(uuid.UUID, self._values.pop(0)))


class _ProgressHeadEngine:
    def __init__(self, conversation_id: str, head_run_id: uuid.UUID) -> None:
        self.connection = _ProgressHeadConnection(conversation_id, head_run_id)

    def begin(self) -> _ConnectionContext:
        return _ConnectionContext(self.connection)


def _ledger(engine: _RecordingEngine) -> PostgresRunLedger:
    return PostgresRunLedger(
        cast(AsyncEngine, engine),
        prompt_version="test",
        retrieval_version="test",
        model_name="test",
    )


def test_agent_run_persists_stream_delivery_and_cancellation_state() -> None:
    expected_columns = {
        "slack_user_id",
        "slack_message_ts",
        "slack_stream_state",
        "slack_stream_mode",
        "slack_stream_ts",
        "last_progress_sequence",
        "delivery_status",
        "delivery_manifest_version",
        "delivery_manifest_hash",
        "cancellation_requested",
    }

    agent_runs = cast(Table, AgentRun.__table__)

    assert expected_columns <= set(agent_runs.columns.keys())


def test_new_baseline_requires_complete_slack_run_identity() -> None:
    agent_runs = cast(Table, AgentRun.__table__)

    assert agent_runs.c.slack_user_id.nullable is False
    assert agent_runs.c.slack_message_ts.nullable is False
    assert "slack_placeholder_ts" not in agent_runs.c
    assert "inngest_event_id" not in agent_runs.c


def test_slack_turns_are_normalized_immutable_queue_rows() -> None:
    slack_turns = cast(Table, SlackTurn.__table__)

    assert tuple(column.name for column in slack_turns.primary_key.columns) == ("event_id",)
    assert {
        "event_id",
        "slack_team_id",
        "slack_channel_id",
        "slack_user_id",
        "slack_message_ts",
        "message_ts_value",
        "slack_thread_ts",
        "conversation_id",
        "kind",
        "status",
        "agent_run_id",
        "created_at",
        "claimed_at",
        "completed_at",
    } == set(slack_turns.columns.keys())
    foreign_key = next(iter(slack_turns.foreign_keys))
    assert foreign_key.target_fullname == "agent_runs.id"
    assert foreign_key.ondelete == "RESTRICT"
    assert {
        "ck_slack_turns_kind",
        "ck_slack_turns_status",
        "ck_slack_turns_message_ts_value",
        "ck_slack_turns_message_ts_matches_value",
        "ck_slack_turns_conversation_identity",
        "ck_slack_turns_lifecycle",
        "ck_slack_turns_claimed_after_created",
        "ck_slack_turns_completed_after_claimed",
        "uq_slack_turns_agent_run_id",
    } <= {constraint.name for constraint in slack_turns.constraints}


def test_slack_turn_queue_has_causal_head_and_single_owner_indexes() -> None:
    slack_turns = cast(Table, SlackTurn.__table__)
    causal_index = next(
        index for index in slack_turns.indexes if index.name == "ix_slack_turns_causal_head"
    )
    processing_index = next(
        index
        for index in slack_turns.indexes
        if index.name == "uq_slack_turns_processing_conversation"
    )

    assert tuple(column.name for column in causal_index.columns) == (
        "conversation_id",
        "message_ts_value",
        "created_at",
        "event_id",
    )
    assert "status IN ('pending', 'processing')" in str(
        causal_index.dialect_options["postgresql"]["where"]
    )
    assert processing_index.unique is True
    assert tuple(column.name for column in processing_index.columns) == ("conversation_id",)
    assert "status = 'processing'" in str(processing_index.dialect_options["postgresql"]["where"])


def test_run_state_invariants_are_enforced_by_database_constraints() -> None:
    agent_runs = cast(Table, AgentRun.__table__)
    constraint_names = {constraint.name for constraint in agent_runs.constraints}

    assert {
        "ck_agent_runs_lifecycle_timestamps",
        "ck_agent_runs_succeeded_state",
        "ck_agent_runs_cancelled_state",
        "ck_agent_runs_error_state",
        "ck_agent_runs_not_started_stream",
        "ck_agent_runs_opening_stream",
        "ck_agent_runs_identified_stream",
        "ck_agent_runs_delivery_has_manifest",
        "ck_agent_runs_cancelled_delivery",
    } <= constraint_names


def test_slack_thread_ownership_lookup_has_a_composite_index() -> None:
    agent_runs = cast(Table, AgentRun.__table__)
    slack_thread_index = next(
        index for index in agent_runs.indexes if index.name == "ix_agent_runs_slack_thread"
    )

    assert tuple(column.name for column in slack_thread_index.columns) == (
        "slack_team_id",
        "slack_channel_id",
        "slack_thread_ts",
    )


def test_active_progress_surface_has_one_database_owner_per_conversation() -> None:
    agent_runs = cast(Table, AgentRun.__table__)
    progress_index = next(
        index
        for index in agent_runs.indexes
        if index.name == "uq_agent_runs_active_progress_conversation"
    )

    assert progress_index.unique is True
    assert tuple(column.name for column in progress_index.columns) == ("conversation_id",)
    predicate = str(progress_index.dialect_options["postgresql"]["where"])
    assert "status IN ('queued', 'running')" in predicate
    assert "slack_stream_state <> 'not_started'" in predicate


async def test_non_head_run_cannot_claim_a_progress_surface() -> None:
    requested_run_id = uuid.uuid4()
    head_run_id = uuid.uuid4()
    engine = _ProgressHeadEngine("T1:C1:1.0", head_run_id)

    claimed = await _ledger(cast(_RecordingEngine, engine)).transition_stream(
        requested_run_id,
        expected_state=SlackStreamState.NOT_STARTED,
        target_state=SlackStreamState.OPENING,
        mode=SlackStreamMode.CHUNKS,
    )

    assert claimed is False
    assert len(engine.connection.statements) == 2
    head_query = str(engine.connection.statements[1])
    assert "agent_runs.conversation_id" in head_query
    assert "agent_runs.status IN" in head_query
    assert "FOR UPDATE" in head_query


def test_delivery_parts_are_ordered_per_run_and_cascade_with_the_run() -> None:
    delivery_parts = cast(Table, RunDeliveryPart.__table__)
    primary_key_columns = tuple(column.name for column in delivery_parts.primary_key.columns)
    foreign_key = next(iter(delivery_parts.foreign_keys))

    assert primary_key_columns == ("agent_run_id", "part_number")
    assert foreign_key.target_fullname == "agent_runs.id"
    assert foreign_key.ondelete == "CASCADE"


def test_slack_stop_event_persists_identity_and_original_outcome() -> None:
    stop_events = cast(Table, SlackStopEvent.__table__)

    assert tuple(column.name for column in stop_events.primary_key.columns) == ("event_id",)
    assert {
        "slack_team_id",
        "slack_channel_id",
        "slack_user_id",
        "slack_thread_ts",
        "slack_event_ts",
        "agent_run_id",
        "accepted",
    } <= set(stop_events.columns.keys())
    foreign_key = next(iter(stop_events.foreign_keys))
    assert foreign_key.target_fullname == "agent_runs.id"
    assert foreign_key.ondelete == "RESTRICT"
    assert "ck_slack_stop_events_accepted_run" in {
        constraint.name for constraint in stop_events.constraints
    }


def test_slack_stopped_streams_are_atomic_ordered_rows() -> None:
    stopped_streams = cast(Table, SlackStoppedStream.__table__)

    assert tuple(column.name for column in stopped_streams.primary_key.columns) == (
        "event_id",
        "stream_order",
    )
    assert "slack_message_ts" in stopped_streams.columns
    foreign_key = next(iter(stopped_streams.foreign_keys))
    assert foreign_key.target_fullname == "slack_stop_events.event_id"
    assert foreign_key.ondelete == "CASCADE"
    assert {
        "ck_slack_stopped_streams_stream_order",
        "uq_slack_stopped_streams_event_timestamp",
    } <= {constraint.name for constraint in stopped_streams.constraints}


async def test_latest_delivered_agent_response_read_is_visible_success_only_and_bounded() -> None:
    response = AgentResponse(
        answer="Which customer?",
        disposition=QuestionDisposition.NEEDS_CLARIFICATION,
    )
    engine = _RecordingEngine(response.model_dump(mode="json"))

    result = await _ledger(engine).get_latest_delivered_agent_response("T1", "C1", "1.0")

    assert result == response

    statement = str(engine.connection.statements[0])
    assert "agent_runs.slack_team_id" in statement
    assert "agent_runs.slack_channel_id" in statement
    assert "agent_runs.slack_thread_ts" in statement
    assert "agent_runs.status IN" in statement
    assert RunStatus.RUNNING.value in str(engine.connection.statements[0].compile().params.values())
    assert RunStatus.SUCCEEDED.value in str(
        engine.connection.statements[0].compile().params.values()
    )
    assert DeliveryStatus.DELIVERED.value in str(
        engine.connection.statements[0].compile().params.values()
    )
    assert "agent_runs.result_json IS NOT NULL" in statement
    assert "ORDER BY CAST(agent_runs.slack_message_ts AS NUMERIC) DESC" in statement
    assert "LIMIT" in statement


async def test_duplicate_stop_replays_original_run_after_a_later_run_exists() -> None:
    original_run_id = uuid.uuid4()
    later_run_id = uuid.uuid4()
    engine = _ReplayEngine(original_run_id, accepted=True)

    claim = await _ledger(cast(_RecordingEngine, engine)).claim_cancellation(
        event_id="EvStop",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        thread_ts="1.0",
        event_ts="3.0",
        streaming_message_timestamps=("2.0",),
    )

    assert claim == CancellationClaim(run_id=original_run_id, accepted=True)
    assert claim.run_id != later_run_id
    # Conflict replay reads only its immutable event mapping; it never selects current runs.
    assert len(engine.connection.statements) == 2
