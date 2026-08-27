from __future__ import annotations

import uuid
from datetime import UTC, datetime

import pytest

from knowledge_assistant.agent.models import AgentResponse
from knowledge_assistant.persistence.models import RunStatus
from knowledge_assistant.persistence.repositories import (
    RunTransitionError,
    _LockedRunState,
    _require_agent_result_readable,
    _require_matching_persisted_result,
    _require_single_run_update,
    _should_persist_agent_result,
    should_apply_run_transition,
)


@pytest.mark.parametrize(
    ("current", "target"),
    [
        (RunStatus.QUEUED, RunStatus.RUNNING),
        (RunStatus.QUEUED, RunStatus.FAILED),
        (RunStatus.RUNNING, RunStatus.SUCCEEDED),
        (RunStatus.RUNNING, RunStatus.FAILED),
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
        (RunStatus.RUNNING, RunStatus.CANCELLED),
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
