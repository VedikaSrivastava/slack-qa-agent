"""Persistence boundary for idempotent agent-run lifecycle updates."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from decimal import Decimal, InvalidOperation
from typing import Any, Protocol

from sqlalchemy import Numeric, cast, delete, func, insert, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.execution.models import QuestionJob
from knowledge_assistant.persistence.models import (
    AgentRun,
    DeliveryStatus,
    RunDeliveryPart,
    RunSource,
    RunStatus,
    SlackStopEvent,
    SlackStoppedStream,
    SlackStreamMode,
    SlackStreamState,
    SlackTurn,
    SlackTurnKind,
    SlackTurnStatus,
)


class RunTransitionError(RuntimeError):
    """Raised when a persisted run receives an illegal lifecycle transition."""


_ALLOWED_RUN_TRANSITIONS = {
    RunStatus.QUEUED: frozenset({RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED}),
    RunStatus.RUNNING: frozenset({RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}),
}
_IDEMPOTENT_TERMINAL_STATUSES = frozenset(
    {RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED}
)

_ALLOWED_STREAM_TRANSITIONS = {
    SlackStreamState.NOT_STARTED: frozenset({SlackStreamState.OPENING, SlackStreamState.DEGRADED}),
    SlackStreamState.OPENING: frozenset(
        {
            SlackStreamState.OPEN,
            SlackStreamState.UNCERTAIN,
            SlackStreamState.DEGRADED,
        }
    ),
    SlackStreamState.OPEN: frozenset(
        {
            SlackStreamState.STOPPING,
            SlackStreamState.UNCERTAIN,
            SlackStreamState.DEGRADED,
        }
    ),
    SlackStreamState.STOPPING: frozenset(
        {
            SlackStreamState.STOPPED,
            SlackStreamState.UNCERTAIN,
            SlackStreamState.DEGRADED,
        }
    ),
    SlackStreamState.UNCERTAIN: frozenset({SlackStreamState.STOPPED, SlackStreamState.DEGRADED}),
}

_ALLOWED_DELIVERY_TRANSITIONS = {
    DeliveryStatus.PENDING: frozenset(
        {DeliveryStatus.DELIVERING, DeliveryStatus.FAILED, DeliveryStatus.CANCELLED}
    ),
    DeliveryStatus.DELIVERING: frozenset(
        {DeliveryStatus.DELIVERED, DeliveryStatus.FAILED, DeliveryStatus.CANCELLED}
    ),
    DeliveryStatus.FAILED: frozenset({DeliveryStatus.DELIVERING, DeliveryStatus.CANCELLED}),
}

_STREAM_TERMINAL_STATES = frozenset(
    {
        SlackStreamState.NOT_STARTED,
        SlackStreamState.STOPPED,
        SlackStreamState.DEGRADED,
    }
)
_TERMINAL_DELIVERY_STATUSES = frozenset({DeliveryStatus.DELIVERED, DeliveryStatus.CANCELLED})
_PROGRESS_SURFACE_STATES = frozenset({SlackStreamState.OPEN})
_ACTIVE_RUN_STATUSES = frozenset({RunStatus.QUEUED, RunStatus.RUNNING})
_CANCELLABLE_DELIVERY_STATUSES = frozenset({DeliveryStatus.PENDING, DeliveryStatus.FAILED})
_SAFE_STOP_FALLBACK_STATES = frozenset({SlackStreamState.UNCERTAIN, SlackStreamState.DEGRADED})
MAX_DELIVERY_PARTS = 100
_ACTIVE_TURN_STATUSES = frozenset({SlackTurnStatus.PENDING, SlackTurnStatus.PROCESSING})
_TERMINAL_TURN_STATUSES = frozenset(
    {
        SlackTurnStatus.ROUTED,
        SlackTurnStatus.SUPPRESSED,
        SlackTurnStatus.FAILED,
    }
)


@dataclass(frozen=True)
class SlackTurnRecord:
    """Typed durable state for one Slack event in the conversation queue."""

    event_id: str
    team_id: str
    channel_id: str
    user_id: str
    message_ts: str
    thread_ts: str
    conversation_id: str
    kind: SlackTurnKind
    status: SlackTurnStatus
    agent_run_id: uuid.UUID | None
    created_at: datetime
    claimed_at: datetime | None
    completed_at: datetime | None


@dataclass(frozen=True)
class SlackTurnEnsureResult:
    turn: SlackTurnRecord
    was_created: bool


@dataclass(frozen=True)
class SlackTurnClaim:
    turn: SlackTurnRecord
    should_process: bool
    was_claimed: bool


@dataclass(frozen=True)
class _TurnClaimDecision:
    should_process: bool
    should_update: bool


@dataclass(frozen=True)
class _SlackTurnIdentity:
    event_id: str
    team_id: str
    channel_id: str
    user_id: str
    message_ts: str
    message_ts_value: Decimal
    thread_ts: str
    conversation_id: str
    kind: SlackTurnKind


def resolve_turn_claim(
    status: SlackTurnStatus,
    *,
    is_causal_head: bool,
) -> _TurnClaimDecision:
    """Authorize only the causal head, while allowing its processing replay."""

    if status in _TERMINAL_TURN_STATUSES:
        return _TurnClaimDecision(should_process=False, should_update=False)
    if status not in _ACTIVE_TURN_STATUSES:
        raise RunTransitionError(f"Unknown Slack turn status: {status}")
    if not is_causal_head:
        return _TurnClaimDecision(should_process=False, should_update=False)
    return _TurnClaimDecision(
        should_process=True,
        should_update=status == SlackTurnStatus.PENDING,
    )


def should_apply_turn_transition(
    current: SlackTurnStatus,
    target: SlackTurnStatus,
    *,
    agent_run_id: uuid.UUID | None,
) -> bool:
    """Validate one terminal turn transition and idempotent replay."""

    if target not in _TERMINAL_TURN_STATUSES:
        raise RunTransitionError(f"Slack turn target is not terminal: {target.value}")
    _require_terminal_turn_link(target, agent_run_id)
    if current == target:
        return False
    if current != SlackTurnStatus.PROCESSING:
        raise RunTransitionError(
            f"Illegal Slack turn transition: {current.value} -> {target.value}"
        )
    return True


def _require_terminal_turn_link(
    target: SlackTurnStatus,
    agent_run_id: uuid.UUID | None,
) -> None:
    if target == SlackTurnStatus.ROUTED and agent_run_id is None:
        raise RunTransitionError("A routed Slack turn must link an agent run")
    if target in {SlackTurnStatus.SUPPRESSED, SlackTurnStatus.FAILED} and agent_run_id is not None:
        raise RunTransitionError(f"A {target.value} Slack turn cannot link an agent run")


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


def should_apply_stream_transition(
    current: SlackStreamState,
    target: SlackStreamState,
) -> bool:
    """Validate a persisted Slack stream transition."""

    if current == target:
        return False
    if target in _ALLOWED_STREAM_TRANSITIONS.get(current, frozenset()):
        return True
    raise RunTransitionError(f"Illegal Slack stream transition: {current.value} -> {target.value}")


def should_apply_delivery_transition(
    current: DeliveryStatus,
    target: DeliveryStatus,
) -> bool:
    """Validate final-answer delivery state and idempotent terminal replays."""

    if current == target:
        return False
    if target in _ALLOWED_DELIVERY_TRANSITIONS.get(current, frozenset()):
        return True
    raise RunTransitionError(f"Illegal delivery transition: {current.value} -> {target.value}")


def should_advance_progress(current_sequence: int, next_sequence: int) -> bool:
    """Accept a newer progress event while making stale retries harmless."""

    if current_sequence < 0:
        raise RunTransitionError("Persisted progress sequence cannot be negative")
    if next_sequence < 1:
        raise RunTransitionError("Progress sequence must be positive")
    return next_sequence > current_sequence


def can_accept_cancellation(status: RunStatus, delivery_status: DeliveryStatus) -> bool:
    """Allow Stop only before any final-answer delivery side effect is in flight."""

    return status in _ACTIVE_RUN_STATUSES and delivery_status in _CANCELLABLE_DELIVERY_STATUSES


def _require_progress_surface(state: SlackStreamState, run_id: uuid.UUID) -> None:
    if state not in _PROGRESS_SURFACE_STATES:
        raise RunTransitionError(
            f"Cannot record progress while Slack progress surface is {state.value}: {run_id}"
        )


def _require_delivery_not_cancelled(
    delivery: _LockedDeliveryState,
    run_id: uuid.UUID,
) -> None:
    """Make a persisted Stop request win before another delivery transition."""

    if delivery.cancellation_requested:
        raise RunTransitionError(
            f"Agent run cannot continue final delivery after cancellation was requested: {run_id}"
        )


def _validate_manifest_inputs(version: int, part_hashes: Sequence[str]) -> tuple[str, ...]:
    if version < 1:
        raise RunTransitionError("Delivery manifest version must be positive")
    hashes = tuple(part_hashes)
    if not hashes:
        raise RunTransitionError("Delivery manifest must contain at least one part")
    if len(hashes) > MAX_DELIVERY_PARTS:
        raise RunTransitionError(f"Delivery manifest exceeds the {MAX_DELIVERY_PARTS}-part limit")
    for content_hash in hashes:
        if (
            len(content_hash) != 64
            or content_hash != content_hash.lower()
            or any(character not in "0123456789abcdef" for character in content_hash)
        ):
            raise RunTransitionError("Delivery part hashes must be lowercase SHA-256 hex digests")
    return hashes


def build_delivery_manifest_hash(version: int, part_hashes: Sequence[str]) -> str:
    """Hash an ordered manifest so a retry cannot silently change delivery content."""

    hashes = _validate_manifest_inputs(version, part_hashes)
    canonical_json = json.dumps(
        {"part_hashes": hashes, "version": version},
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical_json.encode()).hexdigest()


class RunLedger(Protocol):
    async def ensure_turn(
        self,
        *,
        event_id: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        message_ts: str,
        thread_ts: str,
        kind: SlackTurnKind,
    ) -> SlackTurnEnsureResult: ...

    async def claim_turn(self, event_id: str) -> SlackTurnClaim: ...

    async def create_queued_for_turn(
        self,
        job: QuestionJob,
        turn_event_id: str,
    ) -> tuple[uuid.UUID, bool]: ...

    async def complete_turn(
        self,
        event_id: str,
        target: SlackTurnStatus,
    ) -> bool: ...

    async def get_turn(self, event_id: str) -> SlackTurnRecord | None: ...

    async def observe_run(self, run_id: uuid.UUID) -> RunObservation: ...

    async def claim_run(self, run_id: uuid.UUID) -> RunClaim: ...

    async def request_cancellation(self, run_id: uuid.UUID) -> RunObservation: ...

    async def claim_cancellation(
        self,
        *,
        event_id: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        thread_ts: str,
        event_ts: str,
        streaming_message_timestamps: Sequence[str] = (),
    ) -> CancellationClaim: ...

    async def mark_cancelled(self, run_id: uuid.UUID) -> bool: ...

    async def get_latest_delivered_agent_response(
        self,
        team_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> AgentResponse | None: ...

    async def get_persisted_agent_result(self, run_id: uuid.UUID) -> AgentResponse | None: ...

    async def persist_agent_result(self, run_id: uuid.UUID, response: AgentResponse) -> None: ...

    async def mark_succeeded(self, run_id: uuid.UUID, response: AgentResponse) -> None: ...

    async def mark_failed(self, run_id: uuid.UUID, *, code: str, message: str) -> None: ...

    async def get_delivery(self, run_id: uuid.UUID) -> DeliveryState: ...

    async def transition_stream(
        self,
        run_id: uuid.UUID,
        *,
        expected_state: SlackStreamState,
        target_state: SlackStreamState,
        mode: SlackStreamMode | None = None,
        timestamp: str | None = None,
    ) -> bool: ...

    async def acknowledge_stream_open(
        self,
        run_id: uuid.UUID,
        *,
        mode: SlackStreamMode,
        timestamp: str,
    ) -> StreamOpenAcknowledgement: ...

    async def advance_progress(self, run_id: uuid.UUID, sequence: int) -> bool: ...

    async def install_delivery_manifest(
        self,
        run_id: uuid.UUID,
        *,
        version: int,
        part_hashes: Sequence[str],
    ) -> DeliveryManifest: ...

    async def get_delivery_manifest(
        self,
        run_id: uuid.UUID,
    ) -> DeliveryManifest | None: ...

    async def claim_delivery(self, run_id: uuid.UUID) -> bool: ...

    async def acknowledge_delivery_part(
        self,
        run_id: uuid.UUID,
        *,
        part_number: int,
        content_hash: str,
        slack_message_ts: str,
    ) -> bool: ...

    async def mark_delivery_delivered(self, run_id: uuid.UUID) -> bool: ...

    async def mark_delivery_failed(self, run_id: uuid.UUID) -> bool: ...

    async def mark_delivery_cancelled(self, run_id: uuid.UUID) -> bool: ...

    async def set_response(self, run_id: uuid.UUID, timestamp: str) -> None: ...


@dataclass(frozen=True)
class DeliveryState:
    channel_id: str
    thread_ts: str
    response_ts: str | None
    team_id: str
    user_id: str
    stream_state: SlackStreamState = SlackStreamState.NOT_STARTED
    stream_mode: SlackStreamMode | None = None
    stream_ts: str | None = None
    last_progress_sequence: int = 0
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    delivery_manifest_version: int | None = None
    delivery_manifest_hash: str | None = None
    cancellation_requested: bool = False


@dataclass(frozen=True)
class RunObservation:
    status: RunStatus
    cancellation_requested: bool

    @property
    def is_terminal(self) -> bool:
        return self.status in _IDEMPOTENT_TERMINAL_STATUSES


@dataclass(frozen=True)
class RunClaim:
    status: RunStatus
    should_process: bool
    cancellation_requested: bool


@dataclass(frozen=True)
class CancellationClaim:
    """Persisted outcome for one Slack Stop event."""

    run_id: uuid.UUID | None
    accepted: bool


@dataclass(frozen=True)
class StreamOpenAcknowledgement:
    """Durable disposition of a Slack stream that was already created remotely."""

    cancellation_requested: bool
    should_close: bool


@dataclass(frozen=True)
class DeliveryPartState:
    part_number: int
    content_hash: str
    slack_message_ts: str | None
    acknowledged_at: datetime | None


@dataclass(frozen=True)
class DeliveryManifest:
    version: int
    manifest_hash: str
    parts: tuple[DeliveryPartState, ...]


@dataclass(frozen=True)
class _LockedRunState:
    status: RunStatus
    queued_at: datetime
    started_at: datetime | None
    result_json: dict[str, Any] | None
    cancellation_requested: bool
    delivery_status: DeliveryStatus
    stream_state: SlackStreamState


@dataclass(frozen=True)
class _LockedDeliveryState:
    response_ts: str | None
    stream_state: SlackStreamState
    stream_mode: SlackStreamMode | None
    stream_ts: str | None
    last_progress_sequence: int
    delivery_status: DeliveryStatus
    delivery_manifest_version: int | None
    delivery_manifest_hash: str | None
    cancellation_requested: bool


@dataclass(frozen=True)
class _CancellationCandidate:
    run_id: uuid.UUID
    status: RunStatus
    delivery_status: DeliveryStatus
    stream_state: SlackStreamState
    stream_ts: str | None
    message_ts: str
    queued_at: datetime


def _select_cancellation_candidate(
    candidates: Sequence[_CancellationCandidate],
    streaming_message_timestamps: Sequence[str],
    *,
    stop_event_ts: str,
) -> _CancellationCandidate | None:
    """Resolve a Stop to one active run without guessing across known streams."""

    active_candidates = tuple(
        candidate for candidate in candidates if candidate.status in _ACTIVE_RUN_STATUSES
    )
    if not active_candidates:
        return None

    timestamp_set = frozenset(streaming_message_timestamps)
    if timestamp_set:
        exact_candidates = tuple(
            candidate for candidate in active_candidates if candidate.stream_ts in timestamp_set
        )
        if exact_candidates:
            eligible_candidates = exact_candidates
        else:
            # A known, different stream timestamp belongs to another turn. Fallback is
            # limited to runs for which a reliable native stream identity was unavailable.
            eligible_candidates = tuple(
                candidate
                for candidate in active_candidates
                if (
                    candidate.stream_ts is None
                    or candidate.stream_state in _SAFE_STOP_FALLBACK_STATES
                )
                and _candidate_existed_at_stop(candidate, stop_event_ts)
            )
    else:
        eligible_candidates = tuple(
            candidate
            for candidate in active_candidates
            if _candidate_existed_at_stop(candidate, stop_event_ts)
        )

    if not eligible_candidates:
        return None
    return max(
        eligible_candidates,
        key=lambda candidate: (
            candidate.status == RunStatus.RUNNING,
            candidate.queued_at,
            candidate.run_id.int,
        ),
    )


def _candidate_existed_at_stop(
    candidate: _CancellationCandidate,
    stop_event_ts: str,
) -> bool:
    """Fail closed unless both timestamps establish causal Slack ordering."""

    try:
        message_value = _parse_slack_timestamp_value(candidate.message_ts)
        stop_value = _parse_slack_timestamp_value(stop_event_ts)
    except RunTransitionError:
        return False
    return message_value <= stop_value


def _resolve_stream_open_ack_state(
    current_state: SlackStreamState,
    *,
    cancellation_requested: bool,
) -> tuple[SlackStreamState, bool]:
    """Choose a durable state after Slack has already created the remote stream."""

    if current_state == SlackStreamState.NOT_STARTED:
        raise RunTransitionError("Cannot acknowledge a stream that was never started")
    if current_state == SlackStreamState.STOPPED:
        return SlackStreamState.STOPPED, False
    if current_state == SlackStreamState.STOPPING:
        return SlackStreamState.STOPPING, True
    if cancellation_requested or current_state == SlackStreamState.DEGRADED:
        return SlackStreamState.STOPPING, True
    return SlackStreamState.OPEN, False


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
                AgentRun.cancellation_requested,
                AgentRun.delivery_status,
                AgentRun.slack_stream_state,
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
        cancellation_requested=row.cancellation_requested,
        delivery_status=DeliveryStatus(row.delivery_status),
        stream_state=SlackStreamState(row.slack_stream_state),
    )


async def _lock_progress_head(
    connection: AsyncConnection,
    run_id: uuid.UUID,
) -> bool:
    """Lock the oldest active turn so only it may claim a progress surface."""

    conversation_id = (
        await connection.execute(select(AgentRun.conversation_id).where(AgentRun.id == run_id))
    ).scalar_one_or_none()
    if conversation_id is None:
        raise RunTransitionError(f"Agent run does not exist: {run_id}")

    causal_order = cast(AgentRun.slack_message_ts, Numeric())
    head_run_id = (
        await connection.execute(
            select(AgentRun.id)
            .where(
                AgentRun.conversation_id == conversation_id,
                AgentRun.status.in_((RunStatus.QUEUED.value, RunStatus.RUNNING.value)),
            )
            .order_by(causal_order, AgentRun.queued_at, AgentRun.id)
            .limit(1)
            .with_for_update()
        )
    ).scalar_one_or_none()
    return head_run_id == run_id


async def _lock_delivery_for_transition(
    connection: AsyncConnection,
    run_id: uuid.UUID,
) -> _LockedDeliveryState:
    row = (
        await connection.execute(
            select(
                AgentRun.slack_response_ts,
                AgentRun.slack_stream_state,
                AgentRun.slack_stream_mode,
                AgentRun.slack_stream_ts,
                AgentRun.last_progress_sequence,
                AgentRun.delivery_status,
                AgentRun.delivery_manifest_version,
                AgentRun.delivery_manifest_hash,
                AgentRun.cancellation_requested,
            )
            .where(AgentRun.id == run_id)
            .with_for_update()
        )
    ).one_or_none()
    if row is None:
        raise RunTransitionError(f"Agent run does not exist: {run_id}")
    return _LockedDeliveryState(
        response_ts=row.slack_response_ts,
        stream_state=SlackStreamState(row.slack_stream_state),
        stream_mode=(
            SlackStreamMode(row.slack_stream_mode) if row.slack_stream_mode is not None else None
        ),
        stream_ts=row.slack_stream_ts,
        last_progress_sequence=row.last_progress_sequence,
        delivery_status=DeliveryStatus(row.delivery_status),
        delivery_manifest_version=row.delivery_manifest_version,
        delivery_manifest_hash=row.delivery_manifest_hash,
        cancellation_requested=row.cancellation_requested,
    )


async def _load_delivery_manifest(
    connection: AsyncConnection,
    run_id: uuid.UUID,
    *,
    version: int,
    manifest_hash: str,
    lock_parts: bool = False,
) -> DeliveryManifest:
    statement = (
        select(
            RunDeliveryPart.part_number,
            RunDeliveryPart.content_hash,
            RunDeliveryPart.slack_message_ts,
            RunDeliveryPart.acknowledged_at,
        )
        .where(RunDeliveryPart.agent_run_id == run_id)
        .order_by(RunDeliveryPart.part_number)
    )
    if lock_parts:
        statement = statement.with_for_update()
    rows = (await connection.execute(statement)).all()
    return DeliveryManifest(
        version=version,
        manifest_hash=manifest_hash,
        parts=tuple(
            DeliveryPartState(
                part_number=row.part_number,
                content_hash=row.content_hash,
                slack_message_ts=row.slack_message_ts,
                acknowledged_at=row.acknowledged_at,
            )
            for row in rows
        ),
    )


def _require_manifest_identity(
    persisted: DeliveryManifest,
    *,
    version: int,
    manifest_hash: str,
    part_hashes: Sequence[str],
    run_id: uuid.UUID,
) -> None:
    persisted_hashes = tuple(part.content_hash for part in persisted.parts)
    if (
        persisted.version != version
        or persisted.manifest_hash != manifest_hash
        or persisted_hashes != tuple(part_hashes)
    ):
        raise RunTransitionError(f"Agent run already has a different manifest: {run_id}")


def _normalize_slack_turn_identity(
    *,
    event_id: str,
    team_id: str,
    channel_id: str,
    user_id: str,
    message_ts: str,
    thread_ts: str,
    kind: SlackTurnKind,
) -> _SlackTurnIdentity:
    if not isinstance(kind, SlackTurnKind):
        raise RunTransitionError("Slack turn kind must be a SlackTurnKind")
    normalized_event_id = _validate_required_identity(event_id, "Slack event ID", 512)
    normalized_team_id = _validate_required_identity(team_id, "Slack team ID", 128)
    normalized_channel_id = _validate_required_identity(channel_id, "Slack channel ID", 128)
    normalized_user_id = _validate_required_identity(user_id, "Slack user ID", 128)
    normalized_message_ts = _validate_slack_timestamp(message_ts)
    normalized_thread_ts = _validate_slack_timestamp(thread_ts)
    message_ts_value = _parse_slack_timestamp_value(normalized_message_ts)
    _parse_slack_timestamp_value(normalized_thread_ts)
    message_ts_exponent = message_ts_value.as_tuple().exponent
    if not isinstance(message_ts_exponent, int):
        raise RunTransitionError("Slack message timestamp must be finite")
    if message_ts_exponent < -6:
        raise RunTransitionError("Slack message timestamp cannot exceed 6 decimal places")
    if message_ts_value >= Decimal(10) ** 24:
        raise RunTransitionError("Slack message timestamp exceeds the supported range")
    conversation_id = f"{normalized_team_id}:{normalized_channel_id}:{normalized_thread_ts}"
    return _SlackTurnIdentity(
        event_id=normalized_event_id,
        team_id=normalized_team_id,
        channel_id=normalized_channel_id,
        user_id=normalized_user_id,
        message_ts=normalized_message_ts,
        message_ts_value=message_ts_value,
        thread_ts=normalized_thread_ts,
        conversation_id=conversation_id,
        kind=kind,
    )


def _validate_required_identity(value: str, label: str, max_length: int) -> str:
    normalized = value.strip()
    if not normalized or len(normalized) > max_length:
        raise RunTransitionError(f"{label} must contain 1 to {max_length} characters")
    if normalized != value:
        raise RunTransitionError(f"{label} cannot contain surrounding whitespace")
    return normalized


def _turn_record_from_row(row: Any) -> SlackTurnRecord:
    return SlackTurnRecord(
        event_id=row.event_id,
        team_id=row.slack_team_id,
        channel_id=row.slack_channel_id,
        user_id=row.slack_user_id,
        message_ts=row.slack_message_ts,
        thread_ts=row.slack_thread_ts,
        conversation_id=row.conversation_id,
        kind=SlackTurnKind(row.kind),
        status=SlackTurnStatus(row.status),
        agent_run_id=row.agent_run_id,
        created_at=row.created_at,
        claimed_at=row.claimed_at,
        completed_at=row.completed_at,
    )


def _require_matching_turn_identity(
    persisted: SlackTurnRecord,
    expected: _SlackTurnIdentity,
) -> None:
    actual_identity = (
        persisted.event_id,
        persisted.team_id,
        persisted.channel_id,
        persisted.user_id,
        persisted.message_ts,
        persisted.thread_ts,
        persisted.conversation_id,
        persisted.kind,
    )
    expected_identity = (
        expected.event_id,
        expected.team_id,
        expected.channel_id,
        expected.user_id,
        expected.message_ts,
        expected.thread_ts,
        expected.conversation_id,
        expected.kind,
    )
    if actual_identity != expected_identity:
        raise RunTransitionError(f"Slack turn identity is immutable for event {expected.event_id}")


def _require_job_matches_turn(job: QuestionJob, turn: SlackTurnRecord) -> None:
    job_identity = (
        job.event_id,
        job.team_id,
        job.channel_id,
        job.user_id,
        _validate_slack_timestamp(job.message_ts),
        _validate_slack_timestamp(job.thread_ts),
        job.conversation_id,
    )
    turn_identity = (
        turn.event_id,
        turn.team_id,
        turn.channel_id,
        turn.user_id,
        turn.message_ts,
        turn.thread_ts,
        turn.conversation_id,
    )
    if job_identity != turn_identity:
        raise RunTransitionError(f"Question job does not match Slack turn {turn.event_id}")


def _require_job_matches_agent_run(job: QuestionJob, persisted: Any) -> None:
    job_identity = (
        job.agent_run_id,
        job.event_id,
        job.conversation_id,
        job.team_id,
        job.channel_id,
        job.user_id,
        _validate_slack_timestamp(job.message_ts),
        _validate_slack_timestamp(job.thread_ts),
    )
    run_identity = (
        persisted.id,
        persisted.slack_event_id,
        persisted.conversation_id,
        persisted.slack_team_id,
        persisted.slack_channel_id,
        persisted.slack_user_id,
        persisted.slack_message_ts,
        persisted.slack_thread_ts,
    )
    if job_identity != run_identity:
        raise RunTransitionError(f"Agent run identity is immutable for Slack event {job.event_id}")


def _validate_slack_timestamp(timestamp: str) -> str:
    normalized = timestamp.strip()
    if not normalized or len(normalized) > 64:
        raise RunTransitionError("Slack message timestamp must contain 1 to 64 characters")
    return normalized


def _normalize_slack_timestamps(timestamps: Sequence[str]) -> tuple[str, ...]:
    """Validate and deduplicate Slack identities while preserving event order."""

    normalized_timestamps: list[str] = []
    for timestamp in timestamps:
        normalized = _validate_slack_timestamp(timestamp)
        _parse_slack_timestamp_value(normalized)
        normalized_timestamps.append(normalized)
    return tuple(dict.fromkeys(normalized_timestamps))


def _parse_slack_timestamp_value(timestamp: str) -> Decimal:
    normalized = _validate_slack_timestamp(timestamp)
    try:
        value = Decimal(normalized)
    except InvalidOperation as exc:
        raise RunTransitionError("Slack timestamp must be numeric") from exc
    if not value.is_finite() or value < 0:
        raise RunTransitionError("Slack timestamp must be a finite non-negative value")
    return value


def _select_turn_columns() -> tuple[Any, ...]:
    return (
        SlackTurn.event_id,
        SlackTurn.slack_team_id,
        SlackTurn.slack_channel_id,
        SlackTurn.slack_user_id,
        SlackTurn.slack_message_ts,
        SlackTurn.slack_thread_ts,
        SlackTurn.conversation_id,
        SlackTurn.kind,
        SlackTurn.status,
        SlackTurn.agent_run_id,
        SlackTurn.created_at,
        SlackTurn.claimed_at,
        SlackTurn.completed_at,
    )


async def _load_turn_record(
    connection: AsyncConnection,
    event_id: str,
    *,
    lock: bool = False,
) -> SlackTurnRecord | None:
    statement = select(*_select_turn_columns()).where(SlackTurn.event_id == event_id)
    if lock:
        statement = statement.with_for_update()
    row = (await connection.execute(statement)).one_or_none()
    return _turn_record_from_row(row) if row is not None else None


async def _lock_slack_conversation(
    connection: AsyncConnection,
    conversation_id: str,
) -> None:
    """Serialize queue transitions even when a conversation has no stable head row."""

    await connection.execute(
        select(func.pg_advisory_xact_lock(func.hashtextextended(conversation_id, 0)))
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

    async def ensure_turn(
        self,
        *,
        event_id: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        message_ts: str,
        thread_ts: str,
        kind: SlackTurnKind,
    ) -> SlackTurnEnsureResult:
        identity = _normalize_slack_turn_identity(
            event_id=event_id,
            team_id=team_id,
            channel_id=channel_id,
            user_id=user_id,
            message_ts=message_ts,
            thread_ts=thread_ts,
            kind=kind,
        )
        statement = (
            pg_insert(SlackTurn)
            .values(
                event_id=identity.event_id,
                slack_team_id=identity.team_id,
                slack_channel_id=identity.channel_id,
                slack_user_id=identity.user_id,
                slack_message_ts=identity.message_ts,
                message_ts_value=identity.message_ts_value,
                slack_thread_ts=identity.thread_ts,
                conversation_id=identity.conversation_id,
                kind=identity.kind.value,
                status=SlackTurnStatus.PENDING.value,
            )
            .on_conflict_do_nothing(index_elements=[SlackTurn.event_id])
            .returning(SlackTurn.event_id)
        )
        async with self._engine.begin() as connection:
            await _lock_slack_conversation(connection, identity.conversation_id)
            inserted_event_id = (await connection.execute(statement)).scalar_one_or_none()
            turn = await _load_turn_record(connection, identity.event_id, lock=True)
            if turn is None:
                raise RunTransitionError(
                    f"Slack turn disappeared after insert: {identity.event_id}"
                )
            _require_matching_turn_identity(turn, identity)
            return SlackTurnEnsureResult(
                turn=turn,
                was_created=inserted_event_id is not None,
            )

    async def claim_turn(self, event_id: str) -> SlackTurnClaim:
        normalized_event_id = _validate_required_identity(event_id, "Slack event ID", 512)
        async with self._engine.begin() as connection:
            target = await _load_turn_record(connection, normalized_event_id)
            if target is None:
                raise RunTransitionError(f"Slack turn does not exist: {normalized_event_id}")
            await _lock_slack_conversation(connection, target.conversation_id)
            target = await _load_turn_record(connection, normalized_event_id, lock=True)
            if target is None:
                raise RunTransitionError(f"Slack turn does not exist: {normalized_event_id}")
            if target.status in _TERMINAL_TURN_STATUSES:
                # Terminal turns are immutable and should never be reprocessed after delivery.
                return SlackTurnClaim(target, should_process=False, was_claimed=False)

            # A processing row owns the conversation until it reaches a terminal state.
            # This preserves a claimed lease even if an older Slack event arrives late.
            processing_row = (
                await connection.execute(
                    select(*_select_turn_columns())
                    .where(
                        SlackTurn.conversation_id == target.conversation_id,
                        SlackTurn.status == SlackTurnStatus.PROCESSING.value,
                    )
                    .limit(1)
                    .with_for_update()
                )
            ).one_or_none()
            if processing_row is not None:
                processing_turn = _turn_record_from_row(processing_row)
                decision = resolve_turn_claim(
                    target.status,
                    is_causal_head=processing_turn.event_id == normalized_event_id,
                )
                return SlackTurnClaim(
                    target if not decision.should_process else processing_turn,
                    should_process=decision.should_process,
                    was_claimed=False,
                )

            head_row = (
                await connection.execute(
                    select(*_select_turn_columns())
                    .where(
                        SlackTurn.conversation_id == target.conversation_id,
                        SlackTurn.status == SlackTurnStatus.PENDING.value,
                    )
                    .order_by(
                        SlackTurn.message_ts_value,
                        SlackTurn.created_at,
                        SlackTurn.event_id,
                    )
                    .limit(1)
                    .with_for_update()
                )
            ).one_or_none()
            if head_row is None:
                refreshed = await _load_turn_record(
                    connection,
                    normalized_event_id,
                    lock=True,
                )
                if refreshed is None:
                    raise RunTransitionError(
                        f"Slack turn disappeared during claim: {normalized_event_id}"
                    )
                if refreshed.status in _TERMINAL_TURN_STATUSES:
                    return SlackTurnClaim(
                        refreshed,
                        should_process=False,
                        was_claimed=False,
                    )
                raise RunTransitionError(
                    f"Conversation has no causal head for Slack turn {normalized_event_id}"
                )

            head = _turn_record_from_row(head_row)
            decision = resolve_turn_claim(
                target.status,
                is_causal_head=head.event_id == normalized_event_id,
            )
            if not decision.should_update:
                # Non-head candidates stay in wait state until the causal head can advance.
                return SlackTurnClaim(
                    target,
                    should_process=decision.should_process,
                    was_claimed=False,
                )

            claimed_at = datetime.now(UTC)
            result = await connection.execute(
                update(SlackTurn)
                .where(
                    SlackTurn.event_id == normalized_event_id,
                    SlackTurn.status == SlackTurnStatus.PENDING.value,
                )
                .values(
                    status=SlackTurnStatus.PROCESSING.value,
                    claimed_at=claimed_at,
                )
            )
            if result.rowcount != 1:
                raise RunTransitionError(
                    f"Expected to claim one Slack turn {normalized_event_id}, "
                    f"updated {result.rowcount}"
                )
            return SlackTurnClaim(
                replace(
                    head,
                    status=SlackTurnStatus.PROCESSING,
                    claimed_at=claimed_at,
                ),
                should_process=True,
                was_claimed=True,
            )

    async def create_queued_for_turn(
        self,
        job: QuestionJob,
        turn_event_id: str,
    ) -> tuple[uuid.UUID, bool]:
        normalized_event_id = _validate_required_identity(
            turn_event_id,
            "Slack event ID",
            512,
        )
        async with self._engine.begin() as connection:
            turn = await _load_turn_record(connection, normalized_event_id)
            if turn is None:
                raise RunTransitionError(f"Slack turn does not exist: {normalized_event_id}")
            await _lock_slack_conversation(connection, turn.conversation_id)
            turn = await _load_turn_record(connection, normalized_event_id, lock=True)
            if turn is None:
                raise RunTransitionError(f"Slack turn does not exist: {normalized_event_id}")
            if turn.status not in {
                SlackTurnStatus.PROCESSING,
                SlackTurnStatus.ROUTED,
            }:
                # A run can only be created once the turn is queued for processing.
                raise RunTransitionError(
                    f"Cannot create an agent run while Slack turn {normalized_event_id} "
                    f"is {turn.status.value}"
                )
            _require_job_matches_turn(job, turn)
            run_id, was_created = await self._create_queued_in_transaction(connection, job)
            if turn.agent_run_id is not None:
                if turn.agent_run_id != run_id:
                    raise RunTransitionError(
                        f"Agent run link is immutable for Slack turn {normalized_event_id}"
                    )
                return run_id, was_created
            if turn.status == SlackTurnStatus.ROUTED:
                raise RunTransitionError(
                    f"Routed Slack turn has no agent run: {normalized_event_id}"
                )
            result = await connection.execute(
                update(SlackTurn)
                .where(
                    SlackTurn.event_id == normalized_event_id,
                    SlackTurn.status == SlackTurnStatus.PROCESSING.value,
                    SlackTurn.agent_run_id.is_(None),
                )
                .values(agent_run_id=run_id)
            )
            if result.rowcount != 1:
                raise RunTransitionError(
                    f"Expected to link one Slack turn {normalized_event_id}, "
                    f"updated {result.rowcount}"
                )
            return run_id, was_created

    async def complete_turn(
        self,
        event_id: str,
        target: SlackTurnStatus,
    ) -> bool:
        normalized_event_id = _validate_required_identity(event_id, "Slack event ID", 512)
        if not isinstance(target, SlackTurnStatus):
            raise RunTransitionError("Slack turn target must be a SlackTurnStatus")
        async with self._engine.begin() as connection:
            turn = await _load_turn_record(connection, normalized_event_id)
            if turn is None:
                raise RunTransitionError(f"Slack turn does not exist: {normalized_event_id}")
            await _lock_slack_conversation(connection, turn.conversation_id)
            turn = await _load_turn_record(connection, normalized_event_id, lock=True)
            if turn is None:
                raise RunTransitionError(f"Slack turn does not exist: {normalized_event_id}")
            if not should_apply_turn_transition(
                turn.status,
                target,
                agent_run_id=turn.agent_run_id,
            ):
                return False
            result = await connection.execute(
                update(SlackTurn)
                .where(
                    SlackTurn.event_id == normalized_event_id,
                    SlackTurn.status == SlackTurnStatus.PROCESSING.value,
                )
                .values(status=target.value, completed_at=datetime.now(UTC))
            )
            if result.rowcount != 1:
                raise RunTransitionError(
                    f"Expected to complete one Slack turn {normalized_event_id}, "
                    f"updated {result.rowcount}"
                )
            return True

    async def get_turn(self, event_id: str) -> SlackTurnRecord | None:
        normalized_event_id = _validate_required_identity(event_id, "Slack event ID", 512)
        async with self._engine.connect() as connection:
            return await _load_turn_record(connection, normalized_event_id)

    async def _create_queued_in_transaction(
        self,
        connection: AsyncConnection,
        job: QuestionJob,
    ) -> tuple[uuid.UUID, bool]:
        normalized_message_ts = _validate_slack_timestamp(job.message_ts)
        _parse_slack_timestamp_value(normalized_message_ts)
        statement = (
            pg_insert(AgentRun)
            .values(
                id=job.agent_run_id,
                slack_event_id=job.event_id,
                conversation_id=job.conversation_id,
                status=RunStatus.QUEUED.value,
                slack_team_id=job.team_id,
                slack_channel_id=job.channel_id,
                slack_user_id=job.user_id,
                slack_message_ts=normalized_message_ts,
                slack_thread_ts=job.thread_ts,
                prompt_version=self._prompt_version,
                retrieval_version=self._retrieval_version,
                model_name=self._model_name,
            )
            .on_conflict_do_nothing(index_elements=[AgentRun.slack_event_id])
            .returning(AgentRun.id)
        )
        inserted = (await connection.execute(statement)).scalar_one_or_none()
        if inserted is not None:
            return inserted, True
        existing = (
            await connection.execute(
                select(
                    AgentRun.id,
                    AgentRun.slack_event_id,
                    AgentRun.conversation_id,
                    AgentRun.slack_team_id,
                    AgentRun.slack_channel_id,
                    AgentRun.slack_user_id,
                    AgentRun.slack_message_ts,
                    AgentRun.slack_thread_ts,
                ).where(AgentRun.slack_event_id == job.event_id)
            )
        ).one_or_none()
        if existing is None:
            raise RunTransitionError(
                f"Agent run conflict has no matching Slack event: {job.event_id}"
            )
        _require_job_matches_agent_run(job, existing)
        return existing.id, False

    async def observe_run(self, run_id: uuid.UUID) -> RunObservation:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(AgentRun.status, AgentRun.cancellation_requested).where(
                        AgentRun.id == run_id
                    )
                )
            ).one_or_none()
        if row is None:
            raise RunTransitionError(f"Agent run does not exist: {run_id}")
        return RunObservation(
            status=RunStatus(row.status),
            cancellation_requested=row.cancellation_requested,
        )

    async def claim_run(self, run_id: uuid.UUID) -> RunClaim:
        """Start or resume an active run while making terminal retries harmless."""

        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            if run.status in _IDEMPOTENT_TERMINAL_STATUSES:
                return RunClaim(
                    status=run.status,
                    should_process=False,
                    cancellation_requested=run.cancellation_requested,
                )
            if run.cancellation_requested:
                return RunClaim(
                    status=run.status,
                    should_process=False,
                    cancellation_requested=True,
                )
            if run.status == RunStatus.QUEUED:
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
            return RunClaim(
                status=RunStatus.RUNNING,
                should_process=True,
                cancellation_requested=False,
            )

    async def request_cancellation(self, run_id: uuid.UUID) -> RunObservation:
        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            # Stop is accepted only while we still control run lifecycle and before final
            # delivery starts; otherwise this call becomes a non-op for the current state.
            if not can_accept_cancellation(run.status, run.delivery_status):
                return RunObservation(run.status, run.cancellation_requested)
            if not run.cancellation_requested:
                result = await connection.execute(
                    update(AgentRun)
                    .where(AgentRun.id == run_id)
                    .values(cancellation_requested=True)
                )
                _require_single_run_update(result.rowcount, run_id)
            return RunObservation(run.status, True)

    async def claim_cancellation(
        self,
        *,
        event_id: str,
        team_id: str,
        channel_id: str,
        user_id: str,
        thread_ts: str,
        event_ts: str,
        streaming_message_timestamps: Sequence[str] = (),
    ) -> CancellationClaim:
        """Atomically bind one Slack Stop event to one cancellation outcome."""

        normalized_thread_ts = _validate_slack_timestamp(thread_ts)
        normalized_event_ts = _validate_slack_timestamp(event_ts)
        normalized_stream_timestamps = _normalize_slack_timestamps(streaming_message_timestamps)
        insert_event = (
            pg_insert(SlackStopEvent)
            .values(
                event_id=event_id,
                slack_team_id=team_id,
                slack_channel_id=channel_id,
                slack_user_id=user_id,
                slack_thread_ts=normalized_thread_ts,
                slack_event_ts=normalized_event_ts,
                accepted=False,
            )
            .on_conflict_do_nothing(index_elements=[SlackStopEvent.event_id])
            .returning(SlackStopEvent.event_id)
        )
        async with self._engine.begin() as connection:
            inserted_event_id = (await connection.execute(insert_event)).scalar_one_or_none()
            if inserted_event_id is None:
                # A duplicated stop event is treated as idempotent replay, not a rebind.
                replay = (
                    await connection.execute(
                        select(SlackStopEvent.agent_run_id, SlackStopEvent.accepted).where(
                            SlackStopEvent.event_id == event_id
                        )
                    )
                ).one()
                return CancellationClaim(run_id=replay.agent_run_id, accepted=replay.accepted)

            if normalized_stream_timestamps:
                await connection.execute(
                    insert(SlackStoppedStream),
                    [
                        {
                            "event_id": event_id,
                            "stream_order": stream_order,
                            "slack_message_ts": timestamp,
                        }
                        for stream_order, timestamp in enumerate(
                            normalized_stream_timestamps,
                            start=1,
                        )
                    ],
                )

            candidate_rows = (
                await connection.execute(
                    select(
                        AgentRun.id,
                        AgentRun.status,
                        AgentRun.delivery_status,
                        AgentRun.slack_stream_state,
                        AgentRun.slack_stream_ts,
                        AgentRun.slack_message_ts,
                        AgentRun.queued_at,
                    )
                    .where(
                        AgentRun.slack_team_id == team_id,
                        AgentRun.slack_channel_id == channel_id,
                        AgentRun.slack_thread_ts == normalized_thread_ts,
                        AgentRun.status.in_((RunStatus.QUEUED.value, RunStatus.RUNNING.value)),
                    )
                    .with_for_update()
                )
            ).all()
            candidate = _select_cancellation_candidate(
                tuple(
                    _CancellationCandidate(
                        run_id=row.id,
                        status=RunStatus(row.status),
                        delivery_status=DeliveryStatus(row.delivery_status),
                        stream_state=SlackStreamState(row.slack_stream_state),
                        stream_ts=row.slack_stream_ts,
                        message_ts=row.slack_message_ts,
                        queued_at=row.queued_at,
                    )
                    for row in candidate_rows
                ),
                normalized_stream_timestamps,
                stop_event_ts=normalized_event_ts,
            )
            if candidate is None:
                return CancellationClaim(run_id=None, accepted=False)

            accepted = can_accept_cancellation(candidate.status, candidate.delivery_status)
            event_update = await connection.execute(
                update(SlackStopEvent)
                .where(SlackStopEvent.event_id == event_id)
                .values(agent_run_id=candidate.run_id, accepted=accepted)
            )
            if event_update.rowcount != 1:
                raise RunTransitionError(
                    f"Expected one Slack Stop event update, updated {event_update.rowcount}"
                )
            if accepted:
                cancellation_update = await connection.execute(
                    update(AgentRun)
                    .where(AgentRun.id == candidate.run_id)
                    .values(cancellation_requested=True)
                )
                _require_single_run_update(cancellation_update.rowcount, candidate.run_id)
            return CancellationClaim(run_id=candidate.run_id, accepted=accepted)

    async def mark_cancelled(self, run_id: uuid.UUID) -> bool:
        async with self._engine.begin() as connection:
            run = await _lock_run_for_transition(connection, run_id)
            should_update = should_apply_run_transition(run.status, RunStatus.CANCELLED)
            if not should_update:
                return False
            if not run.cancellation_requested:
                raise RunTransitionError(
                    f"Agent run cannot be cancelled without a request: {run_id}"
                )
            if run.stream_state not in _STREAM_TERMINAL_STATES:
                raise RunTransitionError(
                    f"Agent run cannot be cancelled while its Slack stream is "
                    f"{run.stream_state.value}: {run_id}"
                )
            if run.delivery_status == DeliveryStatus.DELIVERED:
                raise RunTransitionError(
                    f"Agent run cannot be cancelled after final delivery: {run_id}"
                )
            now = datetime.now(UTC)
            values: dict[str, Any] = {
                "status": RunStatus.CANCELLED.value,
                "completed_at": now,
                "total_latency_ms": max(
                    0,
                    int((now - run.queued_at).total_seconds() * 1000),
                ),
            }
            if run.delivery_status != DeliveryStatus.CANCELLED:
                values["delivery_status"] = DeliveryStatus.CANCELLED.value
            if run.started_at is not None:
                values["agent_latency_ms"] = max(
                    0,
                    int((now - run.started_at).total_seconds() * 1000),
                )
            result = await connection.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(**values)
            )
            _require_single_run_update(result.rowcount, run_id)
            return True

    async def get_latest_delivered_agent_response(
        self,
        team_id: str,
        channel_id: str,
        thread_ts: str,
    ) -> AgentResponse | None:
        """Return the newest visible response in one thread with a bounded indexed read."""

        normalized_thread_ts = _validate_slack_timestamp(thread_ts)
        async with self._engine.connect() as connection:
            result_json = (
                await connection.execute(
                    select(AgentRun.result_json)
                    .where(
                        AgentRun.slack_team_id == team_id,
                        AgentRun.slack_channel_id == channel_id,
                        AgentRun.slack_thread_ts == normalized_thread_ts,
                        AgentRun.status.in_((RunStatus.RUNNING.value, RunStatus.SUCCEEDED.value)),
                        AgentRun.delivery_status == DeliveryStatus.DELIVERED.value,
                        AgentRun.result_json.is_not(None),
                    )
                    # RUNNING + DELIVERED is the narrow recovery window after Slack accepted the
                    # answer but before Inngest's final ledger acknowledgement committed.
                    .order_by(
                        cast(AgentRun.slack_message_ts, Numeric()).desc(),
                        AgentRun.queued_at.desc(),
                        AgentRun.id.desc(),
                    )
                    .limit(1)
                )
            ).scalar_one_or_none()
        if result_json is None:
            return None
        return AgentResponse.model_validate(result_json)

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
            if run.cancellation_requested:
                raise RunTransitionError(
                    f"Agent run cannot succeed after cancellation was requested: {run_id}"
                )
            if run.delivery_status != DeliveryStatus.DELIVERED:
                raise RunTransitionError(
                    f"Agent run cannot succeed before final delivery: {run_id}"
                )
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
            if run.cancellation_requested:
                raise RunTransitionError(
                    f"Agent run cannot fail after cancellation was requested: {run_id}"
                )
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
                        AgentRun.slack_response_ts,
                        AgentRun.slack_team_id,
                        AgentRun.slack_user_id,
                        AgentRun.slack_stream_state,
                        AgentRun.slack_stream_mode,
                        AgentRun.slack_stream_ts,
                        AgentRun.last_progress_sequence,
                        AgentRun.delivery_status,
                        AgentRun.delivery_manifest_version,
                        AgentRun.delivery_manifest_hash,
                        AgentRun.cancellation_requested,
                    ).where(AgentRun.id == run_id)
                )
            ).one_or_none()
        if row is None:
            raise RunTransitionError(f"Agent run does not exist: {run_id}")
        return DeliveryState(
            channel_id=row.slack_channel_id,
            thread_ts=row.slack_thread_ts,
            response_ts=row.slack_response_ts,
            team_id=row.slack_team_id,
            user_id=row.slack_user_id,
            stream_state=SlackStreamState(row.slack_stream_state),
            stream_mode=(
                SlackStreamMode(row.slack_stream_mode)
                if row.slack_stream_mode is not None
                else None
            ),
            stream_ts=row.slack_stream_ts,
            last_progress_sequence=row.last_progress_sequence,
            delivery_status=DeliveryStatus(row.delivery_status),
            delivery_manifest_version=row.delivery_manifest_version,
            delivery_manifest_hash=row.delivery_manifest_hash,
            cancellation_requested=row.cancellation_requested,
        )

    async def transition_stream(
        self,
        run_id: uuid.UUID,
        *,
        expected_state: SlackStreamState,
        target_state: SlackStreamState,
        mode: SlackStreamMode | None = None,
        timestamp: str | None = None,
    ) -> bool:
        normalized_timestamp = (
            _validate_slack_timestamp(timestamp) if timestamp is not None else None
        )
        async with self._engine.begin() as connection:
            if (
                expected_state == SlackStreamState.NOT_STARTED
                and target_state in {SlackStreamState.OPENING, SlackStreamState.DEGRADED}
                and not await _lock_progress_head(connection, run_id)
            ):
                return False
            delivery = await _lock_delivery_for_transition(connection, run_id)
            if target_state in {SlackStreamState.OPENING, SlackStreamState.OPEN}:
                _require_delivery_not_cancelled(delivery, run_id)
            if delivery.stream_state == target_state:
                if mode is not None and mode != delivery.stream_mode:
                    raise RunTransitionError(f"Slack stream mode changed during a retry: {run_id}")
                if normalized_timestamp is not None and normalized_timestamp != delivery.stream_ts:
                    raise RunTransitionError(
                        f"Slack stream timestamp changed during a retry: {run_id}"
                    )
                return False
            if delivery.stream_state != expected_state:
                raise RunTransitionError(
                    f"Slack stream compare-and-set failed for {run_id}: expected "
                    f"{expected_state.value}, found {delivery.stream_state.value}"
                )
            should_apply_stream_transition(delivery.stream_state, target_state)

            if (
                delivery.stream_mode is not None
                and mode is not None
                and delivery.stream_mode != mode
            ):
                raise RunTransitionError(f"Slack stream mode is immutable: {run_id}")
            if (
                delivery.stream_ts is not None
                and normalized_timestamp is not None
                and delivery.stream_ts != normalized_timestamp
            ):
                raise RunTransitionError(f"Slack stream timestamp is immutable: {run_id}")

            effective_mode = delivery.stream_mode or mode
            effective_timestamp = delivery.stream_ts or normalized_timestamp
            if target_state == SlackStreamState.OPENING and effective_mode is None:
                raise RunTransitionError(f"Opening Slack stream has no mode: {run_id}")
            if target_state == SlackStreamState.OPEN and (
                effective_mode is None or effective_timestamp is None
            ):
                raise RunTransitionError(
                    f"Open Slack stream requires a mode and timestamp: {run_id}"
                )
            if target_state == SlackStreamState.STOPPED and effective_timestamp is None:
                raise RunTransitionError(f"Stopped Slack stream has no timestamp: {run_id}")

            values: dict[str, Any] = {"slack_stream_state": target_state.value}
            if mode is not None:
                values["slack_stream_mode"] = mode.value
            if normalized_timestamp is not None:
                values["slack_stream_ts"] = normalized_timestamp
            result = await connection.execute(
                update(AgentRun).where(AgentRun.id == run_id).values(**values)
            )
            _require_single_run_update(result.rowcount, run_id)
            return True

    async def acknowledge_stream_open(
        self,
        run_id: uuid.UUID,
        *,
        mode: SlackStreamMode,
        timestamp: str,
    ) -> StreamOpenAcknowledgement:
        """Record a completed Slack start even when cancellation won the local race."""

        normalized_timestamp = _validate_slack_timestamp(timestamp)
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            if delivery.stream_mode is not None and delivery.stream_mode != mode:
                raise RunTransitionError(f"Slack stream mode changed during open: {run_id}")
            if delivery.stream_ts is not None and delivery.stream_ts != normalized_timestamp:
                raise RunTransitionError(f"Slack stream timestamp changed during open: {run_id}")

            target_state, should_close = _resolve_stream_open_ack_state(
                delivery.stream_state,
                cancellation_requested=delivery.cancellation_requested,
            )
            values: dict[str, Any] = {}
            if delivery.stream_state != target_state:
                values["slack_stream_state"] = target_state.value
            if delivery.stream_mode is None:
                values["slack_stream_mode"] = mode.value
            if delivery.stream_ts is None:
                values["slack_stream_ts"] = normalized_timestamp
            if values:
                result = await connection.execute(
                    update(AgentRun).where(AgentRun.id == run_id).values(**values)
                )
                _require_single_run_update(result.rowcount, run_id)
            return StreamOpenAcknowledgement(
                cancellation_requested=delivery.cancellation_requested,
                should_close=should_close,
            )

    async def advance_progress(self, run_id: uuid.UUID, sequence: int) -> bool:
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            _require_delivery_not_cancelled(delivery, run_id)
            _require_progress_surface(delivery.stream_state, run_id)
            if not should_advance_progress(delivery.last_progress_sequence, sequence):
                return False
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(last_progress_sequence=sequence)
            )
            _require_single_run_update(result.rowcount, run_id)
            return True

    async def install_delivery_manifest(
        self,
        run_id: uuid.UUID,
        *,
        version: int,
        part_hashes: Sequence[str],
    ) -> DeliveryManifest:
        hashes = _validate_manifest_inputs(version, part_hashes)
        manifest_hash = build_delivery_manifest_hash(version, hashes)
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            _require_delivery_not_cancelled(delivery, run_id)
            if (
                delivery.delivery_manifest_version is not None
                and delivery.delivery_manifest_hash is not None
            ):
                persisted = await _load_delivery_manifest(
                    connection,
                    run_id,
                    version=delivery.delivery_manifest_version,
                    manifest_hash=delivery.delivery_manifest_hash,
                    lock_parts=True,
                )
                _require_manifest_identity(
                    persisted,
                    version=version,
                    manifest_hash=manifest_hash,
                    part_hashes=hashes,
                    run_id=run_id,
                )
                return persisted
            if (
                delivery.delivery_manifest_version is not None
                or delivery.delivery_manifest_hash is not None
            ):
                raise RunTransitionError(f"Agent run has an incomplete manifest: {run_id}")
            if delivery.delivery_status != DeliveryStatus.PENDING:
                raise RunTransitionError(
                    f"Cannot install a delivery manifest while delivery is "
                    f"{delivery.delivery_status.value}: {run_id}"
                )

            await connection.execute(
                insert(RunDeliveryPart),
                [
                    {
                        "agent_run_id": run_id,
                        "part_number": part_number,
                        "content_hash": content_hash,
                    }
                    for part_number, content_hash in enumerate(hashes, start=1)
                ],
            )
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(
                    delivery_manifest_version=version,
                    delivery_manifest_hash=manifest_hash,
                )
            )
            _require_single_run_update(result.rowcount, run_id)
            return DeliveryManifest(
                version=version,
                manifest_hash=manifest_hash,
                parts=tuple(
                    DeliveryPartState(
                        part_number=part_number,
                        content_hash=content_hash,
                        slack_message_ts=None,
                        acknowledged_at=None,
                    )
                    for part_number, content_hash in enumerate(hashes, start=1)
                ),
            )

    async def get_delivery_manifest(
        self,
        run_id: uuid.UUID,
    ) -> DeliveryManifest | None:
        async with self._engine.connect() as connection:
            row = (
                await connection.execute(
                    select(
                        AgentRun.delivery_manifest_version,
                        AgentRun.delivery_manifest_hash,
                    ).where(AgentRun.id == run_id)
                )
            ).one_or_none()
            if row is None:
                raise RunTransitionError(f"Agent run does not exist: {run_id}")
            if row.delivery_manifest_version is None and row.delivery_manifest_hash is None:
                return None
            if row.delivery_manifest_version is None or row.delivery_manifest_hash is None:
                raise RunTransitionError(f"Agent run has an incomplete manifest: {run_id}")
            return await _load_delivery_manifest(
                connection,
                run_id,
                version=row.delivery_manifest_version,
                manifest_hash=row.delivery_manifest_hash,
            )

    async def claim_delivery(self, run_id: uuid.UUID) -> bool:
        """Claim or resume final delivery after a lost workflow acknowledgement."""

        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            _require_delivery_not_cancelled(delivery, run_id)
            if delivery.delivery_manifest_version is None:
                raise RunTransitionError(f"Agent run has no delivery manifest: {run_id}")
            if delivery.delivery_status == DeliveryStatus.DELIVERING:
                return True
            if delivery.delivery_status in _TERMINAL_DELIVERY_STATUSES:
                return False
            should_apply_delivery_transition(
                delivery.delivery_status,
                DeliveryStatus.DELIVERING,
            )
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(delivery_status=DeliveryStatus.DELIVERING.value)
            )
            _require_single_run_update(result.rowcount, run_id)
            return True

    async def acknowledge_delivery_part(
        self,
        run_id: uuid.UUID,
        *,
        part_number: int,
        content_hash: str,
        slack_message_ts: str,
    ) -> bool:
        _validate_manifest_inputs(1, (content_hash,))
        if part_number < 1:
            raise RunTransitionError("Delivery part number must be positive")
        normalized_timestamp = _validate_slack_timestamp(slack_message_ts)
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            _require_delivery_not_cancelled(delivery, run_id)
            row = (
                await connection.execute(
                    select(
                        RunDeliveryPart.content_hash,
                        RunDeliveryPart.slack_message_ts,
                        RunDeliveryPart.acknowledged_at,
                    )
                    .where(
                        RunDeliveryPart.agent_run_id == run_id,
                        RunDeliveryPart.part_number == part_number,
                    )
                    .with_for_update()
                )
            ).one_or_none()
            if row is None:
                raise RunTransitionError(
                    f"Delivery part {part_number} does not exist for run {run_id}"
                )
            if row.content_hash != content_hash:
                raise RunTransitionError(
                    f"Delivery part {part_number} content changed for run {run_id}"
                )
            if row.acknowledged_at is not None:
                if row.slack_message_ts == normalized_timestamp:
                    return False
                raise RunTransitionError(
                    f"Delivery part {part_number} has a conflicting Slack timestamp for "
                    f"run {run_id}"
                )
            if delivery.delivery_status != DeliveryStatus.DELIVERING:
                raise RunTransitionError(
                    f"Cannot acknowledge a part while delivery is "
                    f"{delivery.delivery_status.value}: {run_id}"
                )
            if part_number > 1:
                missing_prior_part = (
                    await connection.execute(
                        select(RunDeliveryPart.part_number)
                        .where(
                            RunDeliveryPart.agent_run_id == run_id,
                            RunDeliveryPart.part_number < part_number,
                            RunDeliveryPart.acknowledged_at.is_(None),
                        )
                        .order_by(RunDeliveryPart.part_number)
                        .limit(1)
                    )
                ).scalar_one_or_none()
                if missing_prior_part is not None:
                    raise RunTransitionError(
                        f"Delivery part {part_number} cannot precede part "
                        f"{missing_prior_part} for run {run_id}"
                    )
            result = await connection.execute(
                update(RunDeliveryPart)
                .where(
                    RunDeliveryPart.agent_run_id == run_id,
                    RunDeliveryPart.part_number == part_number,
                )
                .values(
                    slack_message_ts=normalized_timestamp,
                    acknowledged_at=datetime.now(UTC),
                )
            )
            if result.rowcount != 1:
                raise RunTransitionError(
                    f"Expected to acknowledge delivery part {part_number} for {run_id}, "
                    f"updated {result.rowcount}"
                )
            return True

    async def mark_delivery_delivered(self, run_id: uuid.UUID) -> bool:
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            _require_delivery_not_cancelled(delivery, run_id)
            if not should_apply_delivery_transition(
                delivery.delivery_status,
                DeliveryStatus.DELIVERED,
            ):
                return False
            if (
                delivery.delivery_manifest_version is None
                or delivery.delivery_manifest_hash is None
            ):
                raise RunTransitionError(f"Agent run has no delivery manifest: {run_id}")
            manifest = await _load_delivery_manifest(
                connection,
                run_id,
                version=delivery.delivery_manifest_version,
                manifest_hash=delivery.delivery_manifest_hash,
                lock_parts=True,
            )
            if not manifest.parts or any(part.acknowledged_at is None for part in manifest.parts):
                raise RunTransitionError(f"Agent run has unacknowledged delivery parts: {run_id}")
            return await self._set_delivery_status(
                connection,
                run_id,
                DeliveryStatus.DELIVERED,
            )

    async def mark_delivery_failed(self, run_id: uuid.UUID) -> bool:
        return await self._transition_delivery_status(run_id, DeliveryStatus.FAILED)

    async def mark_delivery_cancelled(self, run_id: uuid.UUID) -> bool:
        return await self._transition_delivery_status(run_id, DeliveryStatus.CANCELLED)

    async def set_response(self, run_id: uuid.UUID, timestamp: str) -> None:
        normalized_timestamp = _validate_slack_timestamp(timestamp)
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            if delivery.response_ts == normalized_timestamp:
                return
            if delivery.response_ts is not None:
                raise RunTransitionError(f"Slack response timestamp is immutable for run {run_id}")
            result = await connection.execute(
                update(AgentRun)
                .where(AgentRun.id == run_id)
                .values(slack_response_ts=normalized_timestamp)
            )
            _require_single_run_update(result.rowcount, run_id)

    async def _transition_delivery_status(
        self,
        run_id: uuid.UUID,
        target: DeliveryStatus,
    ) -> bool:
        async with self._engine.begin() as connection:
            delivery = await _lock_delivery_for_transition(connection, run_id)
            if target == DeliveryStatus.CANCELLED and not delivery.cancellation_requested:
                raise RunTransitionError(
                    f"Agent run cannot cancel delivery without an accepted request: {run_id}"
                )
            if target == DeliveryStatus.FAILED and delivery.cancellation_requested:
                raise RunTransitionError(
                    f"Agent run cannot fail delivery after cancellation was requested: {run_id}"
                )
            if not should_apply_delivery_transition(delivery.delivery_status, target):
                return False
            return await self._set_delivery_status(connection, run_id, target)

    async def _set_delivery_status(
        self,
        connection: AsyncConnection,
        run_id: uuid.UUID,
        target: DeliveryStatus,
    ) -> bool:
        result = await connection.execute(
            update(AgentRun).where(AgentRun.id == run_id).values(delivery_status=target.value)
        )
        _require_single_run_update(result.rowcount, run_id)
        return True
