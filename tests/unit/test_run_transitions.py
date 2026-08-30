from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.persistence.models import (
    DeliveryStatus,
    RunStatus,
    SlackStreamState,
    SlackTurnKind,
    SlackTurnStatus,
)
from knowledge_assistant.persistence.repositories import (
    DeliveryManifest,
    DeliveryPartState,
    RunObservation,
    RunTransitionError,
    SlackTurnRecord,
    _CancellationCandidate,
    _LockedDeliveryState,
    _LockedRunState,
    _normalize_slack_timestamps,
    _normalize_slack_turn_identity,
    _require_agent_result_readable,
    _require_delivery_not_cancelled,
    _require_manifest_identity,
    _require_matching_persisted_result,
    _require_matching_turn_identity,
    _require_progress_surface,
    _require_single_run_update,
    _resolve_stream_open_ack_state,
    _select_cancellation_candidate,
    _should_persist_agent_result,
    build_delivery_manifest_hash,
    can_accept_cancellation,
    resolve_turn_claim,
    should_advance_progress,
    should_apply_delivery_transition,
    should_apply_run_transition,
    should_apply_stream_transition,
    should_apply_turn_transition,
)


def _turn_record(
    *,
    status: SlackTurnStatus = SlackTurnStatus.PENDING,
    user_id: str = "U1",
    agent_run_id: uuid.UUID | None = None,
) -> SlackTurnRecord:
    now = datetime.now(UTC)
    return SlackTurnRecord(
        event_id="Ev1",
        team_id="T1",
        channel_id="C1",
        user_id=user_id,
        message_ts="2.000001",
        thread_ts="1.000001",
        message_text="What changed?",
        conversation_id="T1:C1:1.000001",
        kind=SlackTurnKind.EXPLICIT_MENTION,
        status=status,
        agent_run_id=agent_run_id,
        created_at=now,
        claimed_at=now if status != SlackTurnStatus.PENDING else None,
        completed_at=(
            now
            if status
            in {
                SlackTurnStatus.ROUTED,
                SlackTurnStatus.SUPPRESSED,
                SlackTurnStatus.FAILED,
            }
            else None
        ),
    )


def test_older_processing_turn_blocks_a_newer_pending_turn() -> None:
    decision = resolve_turn_claim(
        SlackTurnStatus.PENDING,
        is_causal_head=False,
    )

    assert decision.should_process is False
    assert decision.should_update is False


def test_processing_turn_replay_remains_authorized_without_second_claim() -> None:
    decision = resolve_turn_claim(
        SlackTurnStatus.PROCESSING,
        is_causal_head=True,
    )

    assert decision.should_process is True
    assert decision.should_update is False


def test_pending_causal_head_is_claimed_once() -> None:
    decision = resolve_turn_claim(
        SlackTurnStatus.PENDING,
        is_causal_head=True,
    )

    assert decision.should_process is True
    assert decision.should_update is True


def test_turn_terminal_transitions_validate_run_link_requirements() -> None:
    run_id = uuid.uuid4()

    assert should_apply_turn_transition(
        SlackTurnStatus.PROCESSING,
        SlackTurnStatus.ROUTED,
        agent_run_id=run_id,
    )
    assert (
        should_apply_turn_transition(
            SlackTurnStatus.ROUTED,
            SlackTurnStatus.ROUTED,
            agent_run_id=run_id,
        )
        is False
    )
    assert should_apply_turn_transition(
        SlackTurnStatus.PROCESSING,
        SlackTurnStatus.SUPPRESSED,
        agent_run_id=None,
    )

    with pytest.raises(RunTransitionError, match="must link an agent run"):
        should_apply_turn_transition(
            SlackTurnStatus.PROCESSING,
            SlackTurnStatus.ROUTED,
            agent_run_id=None,
        )
    with pytest.raises(RunTransitionError, match="cannot link an agent run"):
        should_apply_turn_transition(
            SlackTurnStatus.PROCESSING,
            SlackTurnStatus.FAILED,
            agent_run_id=run_id,
        )
    with pytest.raises(RunTransitionError, match="pending -> suppressed"):
        should_apply_turn_transition(
            SlackTurnStatus.PENDING,
            SlackTurnStatus.SUPPRESSED,
            agent_run_id=None,
        )


def test_duplicate_turn_event_rejects_changed_immutable_identity() -> None:
    expected = _normalize_slack_turn_identity(
        event_id="Ev1",
        team_id="T1",
        channel_id="C1",
        user_id="U1",
        message_ts="2.000001",
        thread_ts="1.000001",
        kind=SlackTurnKind.EXPLICIT_MENTION,
    )

    _require_matching_turn_identity(_turn_record(), expected)
    with pytest.raises(RunTransitionError, match="identity is immutable"):
        _require_matching_turn_identity(_turn_record(user_id="U2"), expected)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.QUEUED, RunStatus.FAILED),
        (RunStatus.QUEUED, RunStatus.CANCELLED),
        (RunStatus.RUNNING, RunStatus.SUCCEEDED),
        (RunStatus.RUNNING, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.CANCELLED),
    ],
)
def test_allowed_run_transitions_require_an_update(current: RunStatus, target: RunStatus) -> None:
    assert should_apply_run_transition(current, target) is True


@pytest.mark.parametrize(
    "status",
    [RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED],
)
def test_terminal_retry_replays_are_idempotent(status: RunStatus) -> None:
    assert should_apply_run_transition(status, status) is False


def test_mark_running_retry_is_idempotent() -> None:
    assert should_apply_run_transition(RunStatus.RUNNING, RunStatus.RUNNING) is False


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.QUEUED, RunStatus.SUCCEEDED),
        (RunStatus.SUCCEEDED, RunStatus.FAILED),
        (RunStatus.SUCCEEDED, RunStatus.RUNNING),
        (RunStatus.FAILED, RunStatus.SUCCEEDED),
        (RunStatus.FAILED, RunStatus.RUNNING),
        (RunStatus.CANCELLED, RunStatus.FAILED),
    ],
)
def test_illegal_run_transitions_raise(current: RunStatus, target: RunStatus) -> None:
    with pytest.raises(
        RunTransitionError,
        match=rf"{current.value} -> {target.value}",
    ):
        should_apply_run_transition(current, target)


def test_missing_run_update_fails_instead_of_silently_succeeding() -> None:
    with pytest.raises(RunTransitionError, match="updated 0"):
        _require_single_run_update(0, uuid.uuid4())


def _locked_run(
    status: RunStatus,
    persisted_response: AgentResponse | None = None,
) -> _LockedRunState:
    now = datetime.now(UTC)
    return _LockedRunState(
        status=status,
        queued_at=now,
        started_at=now if status != RunStatus.QUEUED else None,
        result_json=(
            persisted_response.model_dump(mode="json") if persisted_response is not None else None
        ),
        cancellation_requested=False,
        delivery_status=DeliveryStatus.PENDING,
        stream_state=SlackStreamState.NOT_STARTED,
    )


def test_running_run_accepts_one_agent_result() -> None:
    response = AgentResponse(answer="Grounded answer")

    assert _should_persist_agent_result(_locked_run(RunStatus.RUNNING), uuid.uuid4(), response)


def test_persisted_agent_result_replay_is_idempotent() -> None:
    response = AgentResponse(answer="Grounded answer", model_call_count=4)

    assert (
        _should_persist_agent_result(
            _locked_run(RunStatus.RUNNING, response),
            uuid.uuid4(),
            response,
        )
        is False
    )


def test_conflicting_agent_result_is_rejected_without_overwrite() -> None:
    persisted_response = AgentResponse(answer="First answer")
    conflicting_response = AgentResponse(answer="Different answer")

    with pytest.raises(RunTransitionError, match="already has a different result"):
        _should_persist_agent_result(
            _locked_run(RunStatus.RUNNING, persisted_response),
            uuid.uuid4(),
            conflicting_response,
        )


@pytest.mark.parametrize(
    "status",
    [RunStatus.QUEUED, RunStatus.SUCCEEDED, RunStatus.FAILED, RunStatus.CANCELLED],
)
def test_agent_result_cannot_be_persisted_outside_running_state(status: RunStatus) -> None:
    response = AgentResponse(answer="Persisted answer")

    with pytest.raises(RunTransitionError, match=rf"is {status.value}"):
        _should_persist_agent_result(
            _locked_run(status, response),
            uuid.uuid4(),
            response,
        )


def test_success_requires_the_same_intermediate_agent_result() -> None:
    run_id = uuid.uuid4()
    persisted_response = AgentResponse(answer="First answer")

    _require_matching_persisted_result(
        _locked_run(RunStatus.RUNNING, persisted_response),
        run_id,
        persisted_response,
    )
    with pytest.raises(RunTransitionError, match="changed before completion"):
        _require_matching_persisted_result(
            _locked_run(RunStatus.RUNNING, persisted_response),
            run_id,
            AgentResponse(answer="Different answer"),
        )


def test_success_requires_an_intermediate_agent_result() -> None:
    with pytest.raises(RunTransitionError, match="has no persisted result"):
        _require_matching_persisted_result(
            _locked_run(RunStatus.RUNNING),
            uuid.uuid4(),
            AgentResponse(answer="Answer"),
        )


@pytest.mark.parametrize("status", [RunStatus.RUNNING, RunStatus.SUCCEEDED])
def test_active_or_successful_run_can_reuse_agent_result(status: RunStatus) -> None:
    _require_agent_result_readable(status, uuid.uuid4())


@pytest.mark.parametrize(
    "status",
    [RunStatus.QUEUED, RunStatus.FAILED, RunStatus.CANCELLED],
)
def test_invalid_run_status_cannot_reuse_agent_result(status: RunStatus) -> None:
    with pytest.raises(RunTransitionError, match=rf"is {status.value}"):
        _require_agent_result_readable(status, uuid.uuid4())


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (SlackStreamState.NOT_STARTED, SlackStreamState.OPENING),
        (SlackStreamState.NOT_STARTED, SlackStreamState.DEGRADED),
        (SlackStreamState.OPENING, SlackStreamState.OPEN),
        (SlackStreamState.OPENING, SlackStreamState.UNCERTAIN),
        (SlackStreamState.OPEN, SlackStreamState.STOPPING),
        (SlackStreamState.STOPPING, SlackStreamState.STOPPED),
        (SlackStreamState.UNCERTAIN, SlackStreamState.DEGRADED),
    ],
)
def test_allowed_stream_transitions_require_an_update(
    current: SlackStreamState,
    target: SlackStreamState,
) -> None:
    assert should_apply_stream_transition(current, target) is True


@pytest.mark.parametrize("state", list(SlackStreamState))
def test_stream_transition_replays_are_idempotent(state: SlackStreamState) -> None:
    assert should_apply_stream_transition(state, state) is False


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (SlackStreamState.NOT_STARTED, SlackStreamState.OPEN),
        (SlackStreamState.OPENING, SlackStreamState.STOPPED),
        (SlackStreamState.OPEN, SlackStreamState.STOPPED),
        (SlackStreamState.STOPPED, SlackStreamState.OPENING),
        (SlackStreamState.DEGRADED, SlackStreamState.OPENING),
    ],
)
def test_illegal_stream_transitions_raise(
    current: SlackStreamState,
    target: SlackStreamState,
) -> None:
    with pytest.raises(
        RunTransitionError,
        match=rf"{current.value} -> {target.value}",
    ):
        should_apply_stream_transition(current, target)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (DeliveryStatus.PENDING, DeliveryStatus.DELIVERING),
        (DeliveryStatus.DELIVERING, DeliveryStatus.DELIVERED),
        (DeliveryStatus.DELIVERING, DeliveryStatus.FAILED),
        (DeliveryStatus.FAILED, DeliveryStatus.DELIVERING),
        (DeliveryStatus.FAILED, DeliveryStatus.CANCELLED),
    ],
)
def test_allowed_delivery_transitions_require_an_update(
    current: DeliveryStatus,
    target: DeliveryStatus,
) -> None:
    assert should_apply_delivery_transition(current, target) is True


@pytest.mark.parametrize("status", list(DeliveryStatus))
def test_delivery_transition_replays_are_idempotent(status: DeliveryStatus) -> None:
    assert should_apply_delivery_transition(status, status) is False


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (DeliveryStatus.PENDING, DeliveryStatus.DELIVERED),
        (DeliveryStatus.FAILED, DeliveryStatus.DELIVERED),
        (DeliveryStatus.DELIVERED, DeliveryStatus.DELIVERING),
        (DeliveryStatus.CANCELLED, DeliveryStatus.DELIVERING),
    ],
)
def test_illegal_delivery_transitions_raise(
    current: DeliveryStatus,
    target: DeliveryStatus,
) -> None:
    with pytest.raises(
        RunTransitionError,
        match=rf"{current.value} -> {target.value}",
    ):
        should_apply_delivery_transition(current, target)


def test_progress_sequence_only_advances_monotonically() -> None:
    assert should_advance_progress(0, 1) is True
    assert should_advance_progress(3, 5) is True
    assert should_advance_progress(3, 3) is False
    assert should_advance_progress(3, 2) is False


def test_open_native_surface_accepts_progress() -> None:
    _require_progress_surface(SlackStreamState.OPEN, uuid.uuid4())


@pytest.mark.parametrize(
    "state",
    [
        SlackStreamState.NOT_STARTED,
        SlackStreamState.OPENING,
        SlackStreamState.STOPPING,
        SlackStreamState.STOPPED,
        SlackStreamState.UNCERTAIN,
        SlackStreamState.DEGRADED,
    ],
)
def test_inactive_progress_surfaces_reject_progress(state: SlackStreamState) -> None:
    with pytest.raises(RunTransitionError, match="Slack progress surface"):
        _require_progress_surface(state, uuid.uuid4())


@pytest.mark.parametrize(("current", "candidate"), [(-1, 1), (0, 0), (0, -1)])
def test_invalid_progress_sequences_raise(current: int, candidate: int) -> None:
    with pytest.raises(RunTransitionError, match=r"[Pp]rogress sequence"):
        should_advance_progress(current, candidate)


def test_delivery_manifest_hash_is_deterministic_and_order_sensitive() -> None:
    first_hash = "a" * 64
    second_hash = "b" * 64

    digest = build_delivery_manifest_hash(1, (first_hash, second_hash))

    assert digest == build_delivery_manifest_hash(1, (first_hash, second_hash))
    assert digest != build_delivery_manifest_hash(1, (second_hash, first_hash))
    assert digest != build_delivery_manifest_hash(2, (first_hash, second_hash))


@pytest.mark.parametrize(
    ("version", "part_hashes"),
    [
        (0, ("a" * 64,)),
        (1, ()),
        (1, ("A" * 64,)),
        (1, ("not-a-sha256",)),
    ],
)
def test_invalid_delivery_manifest_inputs_raise(
    version: int,
    part_hashes: tuple[str, ...],
) -> None:
    with pytest.raises(RunTransitionError):
        build_delivery_manifest_hash(version, part_hashes)


def test_manifest_replay_requires_exact_ordered_identity() -> None:
    first_hash = "a" * 64
    second_hash = "b" * 64
    digest = build_delivery_manifest_hash(1, (first_hash, second_hash))
    manifest = DeliveryManifest(
        version=1,
        manifest_hash=digest,
        parts=(
            DeliveryPartState(1, first_hash, None, None),
            DeliveryPartState(2, second_hash, None, None),
        ),
    )
    run_id = uuid.uuid4()

    _require_manifest_identity(
        manifest,
        version=1,
        manifest_hash=digest,
        part_hashes=(first_hash, second_hash),
        run_id=run_id,
    )
    with pytest.raises(RunTransitionError, match="different manifest"):
        _require_manifest_identity(
            manifest,
            version=1,
            manifest_hash=build_delivery_manifest_hash(1, (second_hash, first_hash)),
            part_hashes=(second_hash, first_hash),
            run_id=run_id,
        )


def test_run_observation_reports_terminal_state() -> None:
    assert RunObservation(RunStatus.SUCCEEDED, False).is_terminal is True
    assert RunObservation(RunStatus.RUNNING, True).is_terminal is False


def test_persisted_stop_request_blocks_final_delivery() -> None:
    delivery = _LockedDeliveryState(
        response_ts="2.0",
        stream_state=SlackStreamState.STOPPED,
        stream_mode=None,
        stream_ts="2.0",
        last_progress_sequence=4,
        delivery_status=DeliveryStatus.DELIVERING,
        delivery_manifest_version=1,
        delivery_manifest_hash="a" * 64,
        cancellation_requested=True,
    )

    with pytest.raises(RunTransitionError, match="after cancellation was requested"):
        _require_delivery_not_cancelled(delivery, uuid.uuid4())


def _cancellation_candidate(
    *,
    status: RunStatus,
    queued_second: int,
    stream_ts: str | None,
    message_ts: str = "1.0",
    stream_state: SlackStreamState = SlackStreamState.OPEN,
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING,
) -> _CancellationCandidate:
    return _CancellationCandidate(
        run_id=uuid.uuid4(),
        status=status,
        delivery_status=delivery_status,
        stream_state=stream_state,
        stream_ts=stream_ts,
        message_ts=message_ts,
        queued_at=datetime(2026, 1, 1, 0, 0, queued_second, tzinfo=UTC),
    )


def test_stop_with_stream_timestamp_selects_exact_older_run() -> None:
    stopped_run = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=1,
        stream_ts="2.0",
    )
    later_run = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=2,
        stream_ts=None,
        stream_state=SlackStreamState.NOT_STARTED,
    )

    selected = _select_cancellation_candidate(
        (stopped_run, later_run),
        ("2.0",),
        stop_event_ts="3.0",
    )

    assert selected == stopped_run


def test_stop_without_stream_timestamp_prefers_running_over_newer_queued() -> None:
    running = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=1,
        stream_ts=None,
    )
    newer_queued = _cancellation_candidate(
        status=RunStatus.QUEUED,
        queued_second=2,
        stream_ts=None,
        stream_state=SlackStreamState.NOT_STARTED,
    )

    selected = _select_cancellation_candidate(
        (newer_queued, running),
        (),
        stop_event_ts="3.0",
    )

    assert selected == running


def test_timestamp_mismatch_only_falls_back_to_unidentified_or_degraded_run() -> None:
    different_known_stream = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=3,
        stream_ts="9.0",
    )
    degraded = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=1,
        stream_ts="8.0",
        stream_state=SlackStreamState.DEGRADED,
    )

    selected = _select_cancellation_candidate(
        (different_known_stream, degraded),
        ("2.0",),
        stop_event_ts="3.0",
    )

    assert selected == degraded


def test_timestamp_mismatch_rejects_unrelated_known_stream() -> None:
    different_known_stream = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=1,
        stream_ts="9.0",
    )

    assert (
        _select_cancellation_candidate(
            (different_known_stream,),
            ("2.0",),
            stop_event_ts="3.0",
        )
        is None
    )


def test_delayed_timestamp_less_stop_does_not_bind_to_later_run() -> None:
    later_run = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=2,
        stream_ts=None,
        message_ts="4.0",
        stream_state=SlackStreamState.NOT_STARTED,
    )

    selected = _select_cancellation_candidate(
        (later_run,),
        (),
        stop_event_ts="3.0",
    )

    assert selected is None


def test_timestamp_less_stop_matches_current_run_in_same_slack_clock() -> None:
    current_run = _cancellation_candidate(
        status=RunStatus.RUNNING,
        queued_second=1,
        stream_ts=None,
        message_ts="2.0",
        stream_state=SlackStreamState.OPENING,
    )

    selected = _select_cancellation_candidate(
        (current_run,),
        (),
        stop_event_ts="3.0",
    )

    assert selected == current_run


def test_cancelled_remote_stream_acknowledgement_requires_close() -> None:
    target_state, should_close = _resolve_stream_open_ack_state(
        SlackStreamState.OPENING,
        cancellation_requested=True,
    )

    assert target_state is SlackStreamState.STOPPING
    assert should_close is True


def test_normal_remote_stream_acknowledgement_opens_stream() -> None:
    target_state, should_close = _resolve_stream_open_ack_state(
        SlackStreamState.OPENING,
        cancellation_requested=False,
    )

    assert target_state is SlackStreamState.OPEN
    assert should_close is False


@pytest.mark.parametrize(
    ("delivery_status", "accepted"),
    [
        (DeliveryStatus.PENDING, True),
        (DeliveryStatus.FAILED, True),
        (DeliveryStatus.DELIVERING, False),
        (DeliveryStatus.DELIVERED, False),
        (DeliveryStatus.CANCELLED, False),
    ],
)
def test_stop_acceptance_ends_before_delivery_starts(
    delivery_status: DeliveryStatus,
    accepted: bool,
) -> None:
    assert can_accept_cancellation(RunStatus.RUNNING, delivery_status) is accepted


def test_slack_stop_stream_timestamps_are_validated_and_deduplicated_in_order() -> None:
    assert _normalize_slack_timestamps(("2.0", "3.0", "2.0")) == ("2.0", "3.0")

    with pytest.raises(RunTransitionError, match="Slack timestamp"):
        _normalize_slack_timestamps(("not-a-timestamp",))
