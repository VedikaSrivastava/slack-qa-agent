from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import Any, cast
from uuid import UUID, uuid4

import pytest
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient
from slack_sdk.web.slack_response import SlackResponse

from knowledge_assistant.agent.models import (
    AgentResponse,
    EvidenceReference,
    ProgressEvent,
    ProgressStage,
    QuestionDisposition,
)
from knowledge_assistant.integrations.slack.publisher import (
    MAX_SLACK_TEXT,
    MAX_STREAM_MARKDOWN,
    ProgressSurfaceAction,
    SlackDeliveryRejectedError,
    SlackPublisher,
)
from knowledge_assistant.persistence.models import (
    DeliveryStatus,
    SlackStreamMode,
    SlackStreamState,
)
from knowledge_assistant.persistence.repositories import (
    DeliveryManifest,
    DeliveryPartState,
    DeliveryState,
    RunLedger,
    RunTransitionError,
    StreamOpenAcknowledgement,
    build_delivery_manifest_hash,
)


class FakeLedger:
    def __init__(self) -> None:
        self.delivery = DeliveryState(
            channel_id="C1",
            thread_ts="1.0",
            response_ts=None,
            team_id="T1",
            user_id="U1",
        )
        self.manifest: DeliveryManifest | None = None

    async def get_delivery(self, _run_id: UUID) -> DeliveryState:
        return self.delivery

    async def transition_stream(
        self,
        _run_id: UUID,
        *,
        expected_state: SlackStreamState,
        target_state: SlackStreamState,
        mode: SlackStreamMode | None = None,
        timestamp: str | None = None,
    ) -> bool:
        if self.delivery.stream_state == target_state:
            return False
        assert self.delivery.stream_state == expected_state
        self.delivery = replace(
            self.delivery,
            stream_state=target_state,
            stream_mode=self.delivery.stream_mode or mode,
            stream_ts=self.delivery.stream_ts or timestamp,
        )
        return True

    async def advance_progress(self, _run_id: UUID, sequence: int) -> bool:
        if sequence <= self.delivery.last_progress_sequence:
            return False
        self.delivery = replace(self.delivery, last_progress_sequence=sequence)
        return True

    async def acknowledge_stream_open(
        self,
        _run_id: UUID,
        *,
        mode: SlackStreamMode,
        timestamp: str,
    ) -> StreamOpenAcknowledgement:
        should_close = (
            self.delivery.cancellation_requested
            or self.delivery.stream_state is SlackStreamState.DEGRADED
        )
        self.delivery = replace(
            self.delivery,
            stream_state=(SlackStreamState.STOPPING if should_close else SlackStreamState.OPEN),
            stream_mode=self.delivery.stream_mode or mode,
            stream_ts=self.delivery.stream_ts or timestamp,
        )
        return StreamOpenAcknowledgement(
            cancellation_requested=self.delivery.cancellation_requested,
            should_close=should_close,
        )

    async def install_delivery_manifest(
        self,
        _run_id: UUID,
        *,
        version: int,
        part_hashes: tuple[str, ...],
    ) -> DeliveryManifest:
        if self.manifest is None:
            self.manifest = DeliveryManifest(
                version=version,
                manifest_hash=build_delivery_manifest_hash(version, part_hashes),
                parts=tuple(
                    DeliveryPartState(index, content_hash, None, None)
                    for index, content_hash in enumerate(part_hashes, start=1)
                ),
            )
            self.delivery = replace(
                self.delivery,
                delivery_manifest_version=version,
                delivery_manifest_hash=self.manifest.manifest_hash,
            )
        else:
            assert tuple(part.content_hash for part in self.manifest.parts) == part_hashes
        return self.manifest

    async def get_delivery_manifest(self, _run_id: UUID) -> DeliveryManifest | None:
        return self.manifest

    async def claim_delivery(self, _run_id: UUID) -> bool:
        if self.delivery.delivery_status in {DeliveryStatus.DELIVERED, DeliveryStatus.CANCELLED}:
            return False
        self.delivery = replace(self.delivery, delivery_status=DeliveryStatus.DELIVERING)
        return True

    async def acknowledge_delivery_part(
        self,
        _run_id: UUID,
        *,
        part_number: int,
        content_hash: str,
        slack_message_ts: str,
    ) -> bool:
        assert self.manifest is not None
        parts = list(self.manifest.parts)
        current = parts[part_number - 1]
        assert current.content_hash == content_hash
        if current.slack_message_ts is not None:
            assert current.slack_message_ts == slack_message_ts
            return False
        parts[part_number - 1] = replace(
            current,
            slack_message_ts=slack_message_ts,
            acknowledged_at=datetime.now(UTC),
        )
        self.manifest = replace(self.manifest, parts=tuple(parts))
        return True

    async def mark_delivery_delivered(self, _run_id: UUID) -> bool:
        assert self.manifest is not None
        assert all(part.slack_message_ts for part in self.manifest.parts)
        if self.delivery.delivery_status == DeliveryStatus.DELIVERED:
            return False
        self.delivery = replace(self.delivery, delivery_status=DeliveryStatus.DELIVERED)
        return True

    async def mark_delivery_failed(self, _run_id: UUID) -> bool:
        if self.delivery.delivery_status == DeliveryStatus.FAILED:
            return False
        self.delivery = replace(self.delivery, delivery_status=DeliveryStatus.FAILED)
        return True

    async def mark_delivery_cancelled(self, _run_id: UUID) -> bool:
        if self.delivery.delivery_status == DeliveryStatus.CANCELLED:
            return False
        self.delivery = replace(self.delivery, delivery_status=DeliveryStatus.CANCELLED)
        return True

    async def set_response(self, _run_id: UUID, timestamp: str) -> None:
        self.delivery = replace(self.delivery, response_ts=timestamp)


class DeferredProgressLedger(FakeLedger):
    async def transition_stream(
        self,
        run_id: UUID,
        *,
        expected_state: SlackStreamState,
        target_state: SlackStreamState,
        mode: SlackStreamMode | None = None,
        timestamp: str | None = None,
    ) -> bool:
        if expected_state == SlackStreamState.NOT_STARTED and target_state in {
            SlackStreamState.OPENING,
            SlackStreamState.DEGRADED,
        }:
            return False
        return await super().transition_stream(
            run_id,
            expected_state=expected_state,
            target_state=target_state,
            mode=mode,
            timestamp=timestamp,
        )


class OpeningThenOpenLedger(FakeLedger):
    def __init__(self) -> None:
        super().__init__()
        self.delivery = replace(
            self.delivery,
            stream_state=SlackStreamState.OPENING,
            stream_mode=SlackStreamMode.CHUNKS,
        )
        self.read_count = 0

    async def get_delivery(self, _run_id: UUID) -> DeliveryState:
        self.read_count += 1
        if self.read_count >= 3:
            self.delivery = replace(
                self.delivery,
                stream_state=SlackStreamState.OPEN,
                stream_ts="2.0",
            )
        return self.delivery


class OpeningThenCancelledStreamLedger(FakeLedger):
    def __init__(self) -> None:
        super().__init__()
        self.delivery = replace(
            self.delivery,
            stream_state=SlackStreamState.OPENING,
            stream_mode=SlackStreamMode.CHUNKS,
            cancellation_requested=True,
        )
        self.read_count = 0

    async def get_delivery(self, _run_id: UUID) -> DeliveryState:
        self.read_count += 1
        if self.read_count == 2:
            self.delivery = replace(
                self.delivery,
                stream_state=SlackStreamState.STOPPING,
                stream_ts="2.0",
            )
        return self.delivery


class FailFirstOpenAcknowledgementLedger(FakeLedger):
    def __init__(self) -> None:
        super().__init__()
        self.acknowledgement_attempts = 0

    async def acknowledge_stream_open(
        self,
        run_id: UUID,
        *,
        mode: SlackStreamMode,
        timestamp: str,
    ) -> StreamOpenAcknowledgement:
        self.acknowledgement_attempts += 1
        if self.acknowledgement_attempts == 1:
            raise RuntimeError("temporary database failure")
        return await super().acknowledge_stream_open(
            run_id,
            mode=mode,
            timestamp=timestamp,
        )


class FakeSlackClient:
    def __init__(self, *, fail_first_stop: bool = False) -> None:
        self.starts: list[dict[str, Any]] = []
        self.appends: list[dict[str, Any]] = []
        self.stops: list[dict[str, Any]] = []
        self.posts: list[dict[str, Any]] = []
        self.updates: list[dict[str, Any]] = []
        self.api_calls: list[tuple[str, dict[str, Any]]] = []
        self.messages: dict[str, dict[str, Any]] = {}
        self.fail_first_stop = fail_first_stop

    async def chat_startStream(self, **kwargs: Any) -> dict[str, str]:
        self.starts.append(kwargs)
        self.messages["2.0"] = {"ts": "2.0"}
        return {"ts": "2.0"}

    async def chat_appendStream(self, **kwargs: Any) -> dict[str, bool]:
        self.appends.append(kwargs)
        return {"ok": True}

    async def chat_stopStream(self, **kwargs: Any) -> dict[str, bool]:
        self.stops.append(kwargs)
        if self.fail_first_stop and len(self.stops) == 1:
            raise RuntimeError("ambiguous stop")
        timestamp = str(kwargs["ts"])
        self.messages[timestamp] = {
            "ts": timestamp,
            "metadata": kwargs.get("metadata"),
        }
        return {"ok": True}

    async def chat_postMessage(self, **kwargs: Any) -> dict[str, str]:
        self.posts.append(kwargs)
        timestamp = f"3.{len(self.posts)}"
        self.messages[timestamp] = {
            "ts": timestamp,
            "metadata": kwargs.get("metadata"),
        }
        return {"ts": timestamp}

    async def conversations_replies(self, **_kwargs: Any) -> dict[str, object]:
        return {"messages": list(self.messages.values())}

    async def chat_update(self, **kwargs: Any) -> dict[str, bool]:
        self.updates.append(kwargs)
        return {"ok": True}

    async def api_call(self, method: str, **kwargs: Any) -> dict[str, bool]:
        self.api_calls.append((method, kwargs))
        return {"ok": True}


class StopWinsDuringStartClient(FakeSlackClient):
    def __init__(self, ledger: FakeLedger) -> None:
        super().__init__()
        self._ledger = ledger

    async def chat_startStream(self, **kwargs: Any) -> dict[str, str]:
        result = await super().chat_startStream(**kwargs)
        self._ledger.delivery = replace(
            self._ledger.delivery,
            cancellation_requested=True,
        )
        return result


class FailingStartClient(FakeSlackClient):
    async def chat_startStream(self, **kwargs: Any) -> dict[str, str]:
        self.starts.append(kwargs)
        raise RuntimeError("ambiguous stream start")


def _already_stopped_error() -> SlackApiError:
    # Slack SDK's exception constructor is untyped.
    return SlackApiError(  # type: ignore[no-untyped-call]
        "stream already stopped",
        SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.stopStream",
            req_args={},
            data={"ok": False, "error": "message_not_in_streaming_state"},
            headers={},
            status_code=200,
        ),
    )


def _delivery_rejected_error() -> SlackApiError:
    # Slack SDK's exception constructor is untyped.
    return SlackApiError(  # type: ignore[no-untyped-call]
        "delivery rejected",
        SlackResponse(
            client=None,
            http_verb="POST",
            api_url="https://slack.com/api/chat.stopStream",
            req_args={},
            data={"ok": False, "error": "invalid_blocks"},
            headers={},
            status_code=200,
        ),
    )


class UserStoppedStreamClient(FakeSlackClient):
    async def chat_stopStream(self, **kwargs: Any) -> dict[str, bool]:
        self.stops.append(kwargs)
        raise _already_stopped_error()


class LostStopAcknowledgementClient(FakeSlackClient):
    async def chat_stopStream(self, **kwargs: Any) -> dict[str, bool]:
        self.stops.append(kwargs)
        if len(self.stops) == 1:
            timestamp = str(kwargs["ts"])
            self.messages[timestamp] = {
                "ts": timestamp,
                "metadata": kwargs.get("metadata"),
            }
            raise RuntimeError("lost stop response")
        raise _already_stopped_error()


class LaggingStopReadbackClient(LostStopAcknowledgementClient):
    def __init__(self) -> None:
        super().__init__()
        self.reply_reads = 0

    async def conversations_replies(self, **kwargs: Any) -> dict[str, object]:
        self.reply_reads += 1
        if self.reply_reads < 3:
            return {"messages": [{"ts": "2.0"}]}
        return await super().conversations_replies(**kwargs)


class UnknownStopReadbackClient(LostStopAcknowledgementClient):
    async def conversations_replies(self, **_kwargs: Any) -> dict[str, object]:
        raise RuntimeError("temporary Slack read failure")


class DefinitiveDeliveryRejectionClient(FakeSlackClient):
    async def chat_stopStream(self, **kwargs: Any) -> dict[str, bool]:
        self.stops.append(kwargs)
        raise _delivery_rejected_error()


class InterleavedCancellationStopClient(FakeSlackClient):
    def __init__(self) -> None:
        super().__init__()
        self.on_terminal_stop: Callable[[], Awaitable[None]] | None = None
        self.is_stopped = False

    async def chat_stopStream(self, **kwargs: Any) -> dict[str, bool]:
        self.stops.append(kwargs)
        if self.is_stopped:
            raise _already_stopped_error()
        self.is_stopped = True
        timestamp = str(kwargs["ts"])
        self.messages[timestamp] = {
            "ts": timestamp,
            "metadata": kwargs.get("metadata"),
        }
        callback = self.on_terminal_stop
        if "chunks" in kwargs and callback is not None:
            await callback()
        return {"ok": True}


def _publisher(client: FakeSlackClient, ledger: FakeLedger) -> SlackPublisher:
    return SlackPublisher(
        cast(AsyncWebClient, client),
        cast(RunLedger, ledger),
    )


async def _open_surface(publisher: SlackPublisher, run_id: UUID) -> str | None:
    claim = await publisher.claim_progress_surface(run_id)
    if claim.action is ProgressSurfaceAction.READY:
        return claim.timestamp
    if claim.action is not ProgressSurfaceAction.START:
        return None
    timestamp = await publisher.start_claimed_stream(run_id)
    return await publisher.finish_progress_surface(run_id, timestamp)


async def _deliver(
    publisher: SlackPublisher,
    ledger: FakeLedger,
    run_id: UUID,
    response: AgentResponse,
) -> str:
    prepared = await publisher.prepare_delivery(run_id, response)
    if not await publisher.begin_delivery(run_id):
        assert ledger.delivery.response_ts is not None
        return ledger.delivery.response_ts
    timestamps = [
        await publisher.publish_delivery_part(run_id, prepared, part_number)
        for part_number in range(1, len(prepared.parts) + 1)
    ]
    await publisher.complete_delivery(run_id)
    return timestamps[0]


def _progress(run_id: UUID, sequence: int, stage: ProgressStage) -> ProgressEvent:
    return ProgressEvent(
        agent_run_id=str(run_id),
        event_id=f"{run_id}:{sequence}",
        sequence=sequence,
        stage=stage,
    )


async def test_native_stream_shows_sanitized_progress_then_verified_answer() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    first_timestamp = await _open_surface(publisher, run_id)
    retry_timestamp = await _open_surface(publisher, run_id)
    await publisher.publish_progress(run_id, _progress(run_id, 1, ProgressStage.SEARCHING))
    response = AgentResponse(
        answer="Grounded answer [art_a1].",
        sources=[EvidenceReference(artifact_id="art_a1", title="Runbook")],
    )
    prepared = await publisher.prepare_delivery(run_id, response)
    assert await publisher.begin_delivery(run_id) is True
    await publisher.publish_delivery_part(run_id, prepared, 1)
    await publisher.complete_delivery(run_id)

    assert first_timestamp == retry_timestamp == "2.0"
    assert len(client.starts) == 1
    assert client.starts[0]["recipient_team_id"] == "T1"
    assert client.starts[0]["recipient_user_id"] == "U1"
    assert client.starts[0]["task_display_mode"] == "plan"
    assert client.starts[0]["chunks"][0]["title"] == "Understanding the request"
    assert client.starts[0]["chunks"][1]["title"] == "Understanding the request"
    assert len(client.appends) == 1
    prior_chunk, active_chunk, plan_chunk = client.appends[0]["chunks"]
    assert prior_chunk["title"] == "Understanding the request"
    assert prior_chunk["status"] == "complete"
    assert active_chunk["title"] == "Searching company knowledge"
    assert active_chunk["status"] == "in_progress"
    assert plan_chunk == {"type": "plan_update", "title": "Searching company knowledge"}
    rendered_answer = client.stops[0]["chunks"][-1]["text"]
    assert rendered_answer == "Grounded answer."
    assert "Sources" not in rendered_answer


async def test_completed_stream_shows_a_stable_elapsed_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("knowledge_assistant.integrations.slack.publisher.time.time", lambda: 42)
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    await _open_surface(publisher, run_id)
    await _deliver(publisher, ledger, run_id, AgentResponse(answer="Grounded answer"))

    assert client.stops[0]["chunks"][2] == {
        "type": "plan_update",
        "title": "Answered in 40s",
    }
    assert len(client.stops) == 1
    assert client.stops[0]["session_status"] == "active"
    final_chunks = client.stops[0]["chunks"]
    assert final_chunks[0]["type"] == "task_update"
    assert final_chunks[0]["title"] == "Understanding the request"
    assert final_chunks[0]["status"] == "complete"
    assert final_chunks[0]["hide_title"] is False
    assert final_chunks[1]["type"] == "task_update"
    assert final_chunks[1]["title"] == "Answer ready"
    assert final_chunks[1]["status"] == "complete"
    assert final_chunks[1]["hide_title"] is False
    assert final_chunks[-1]["type"] == "markdown_text"
    assert "Grounded answer" in final_chunks[-1]["text"]
    assert client.posts == []
    assert client.updates == []
    assert ledger.delivery.stream_state is SlackStreamState.STOPPED
    assert ledger.delivery.delivery_status is DeliveryStatus.DELIVERED


async def test_completed_plan_keeps_the_last_stage_and_shows_answer_ready() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    await _open_surface(publisher, run_id)
    assert await publisher.publish_progress(run_id, _progress(run_id, 60, ProgressStage.DRAFTING))
    await _deliver(publisher, ledger, run_id, AgentResponse(answer="Grounded answer"))

    task_chunks = [chunk for chunk in client.stops[0]["chunks"] if chunk["type"] == "task_update"]
    assert task_chunks[0]["title"] == "Drafting a grounded answer"
    assert task_chunks[0]["status"] == "complete"
    assert task_chunks[0]["hide_title"] is False
    assert task_chunks[1]["title"] == "Answer ready"
    assert task_chunks[1]["status"] == "complete"
    assert task_chunks[1]["hide_title"] is False
    assert task_chunks[0]["id"] != task_chunks[1]["id"]


async def test_progress_promotes_the_newest_stage_and_retains_prior_steps() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    assert await publisher.publish_progress(run_id, _progress(run_id, 10, ProgressStage.THINKING))
    assert await publisher.publish_progress(run_id, _progress(run_id, 20, ProgressStage.SEARCHING))

    first_prior, first_active, first_plan = client.appends[0]["chunks"]
    assert first_prior["status"] == "complete"
    assert first_active["status"] == "in_progress"
    assert first_active["title"] == "Understanding the question"
    assert first_plan == {"type": "plan_update", "title": "Understanding the question"}
    second_prior, second_active, second_plan = client.appends[1]["chunks"]
    assert second_prior["status"] == "complete"
    assert second_prior["title"] == "Understanding the question"
    assert second_active["status"] == "in_progress"
    assert second_active["title"] == "Searching company knowledge"
    assert second_plan == {"type": "plan_update", "title": "Searching company knowledge"}


async def test_stream_acknowledgement_retry_reuses_persisted_remote_timestamp() -> None:
    ledger = FailFirstOpenAcknowledgementLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    claim = await publisher.claim_progress_surface(run_id)
    assert claim.action is ProgressSurfaceAction.START
    timestamp = await publisher.start_claimed_stream(run_id)
    assert timestamp == "2.0"

    with pytest.raises(RuntimeError, match="temporary database failure"):
        await publisher.finish_progress_surface(run_id, timestamp)
    retry_timestamp = await publisher.finish_progress_surface(run_id, timestamp)

    assert retry_timestamp == "2.0"
    assert len(client.starts) == 1
    assert ledger.acknowledgement_attempts == 2
    assert ledger.delivery.stream_state is SlackStreamState.OPEN


async def test_stop_winning_after_remote_start_closes_the_persisted_stream() -> None:
    ledger = FakeLedger()
    client = StopWinsDuringStartClient(ledger)
    publisher = _publisher(client, ledger)

    timestamp = await _open_surface(publisher, uuid4())

    assert timestamp is None
    assert ledger.delivery.cancellation_requested is True
    assert ledger.delivery.stream_ts == "2.0"
    assert ledger.delivery.stream_state is SlackStreamState.STOPPING
    assert ledger.delivery.response_ts is None
    assert len(client.starts) == 1
    assert len(client.stops) == 1
    assert "chunks" not in client.stops[0]
    assert client.posts == []
    assert client.updates == []
    assert client.api_calls == []


async def test_cancellation_cleanup_waits_for_inflight_stream_identity() -> None:
    ledger = OpeningThenCancelledStreamLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)

    await publisher.publish_cancelled(uuid4())

    assert len(client.stops) == 1
    assert client.stops[0]["ts"] == "2.0"
    assert client.stops[0]["chunks"][0]["title"] == "Work stopped"
    assert client.stops[0]["chunks"][0]["status"] == "error"
    assert client.stops[0]["chunks"][0]["hide_title"] is False
    assert client.stops[0]["chunks"][-1]["text"] == "Stopped at your request."
    assert client.posts == []
    assert client.updates == []
    assert ledger.delivery.stream_state is SlackStreamState.STOPPED
    assert ledger.delivery.delivery_status is DeliveryStatus.CANCELLED


async def test_stale_progress_event_is_not_appended_twice() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    assert await publisher.publish_progress(run_id, _progress(run_id, 2, ProgressStage.VERIFYING))
    assert not await publisher.publish_progress(
        run_id, _progress(run_id, 1, ProgressStage.SEARCHING)
    )

    assert len(client.appends) == 1


async def test_concurrent_worker_defers_to_native_stream_owner() -> None:
    ledger = OpeningThenOpenLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)

    timestamp = await _open_surface(publisher, uuid4())

    assert timestamp is None
    assert client.starts == []
    assert client.posts == []


async def test_stream_open_failure_suppresses_progress_and_posts_only_final_answer() -> None:
    ledger = FakeLedger()
    client = FailingStartClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    assert await _open_surface(publisher, run_id) is None
    assert not await publisher.publish_progress(
        run_id,
        _progress(run_id, 1, ProgressStage.REVIEWING),
    )
    await _deliver(publisher, ledger, run_id, AgentResponse(answer="A concise answer."))

    assert len(client.starts) == 1
    assert len(client.posts) == 1
    assert client.posts[0]["blocks"][0]["text"] == "A concise answer."
    assert client.updates == []
    statuses = [call["json"]["status"] for method, call in client.api_calls]
    assert statuses == ["processing", "active"]
    assert ledger.delivery.stream_state is SlackStreamState.DEGRADED
    assert ledger.delivery.delivery_status is DeliveryStatus.DELIVERED


async def test_clarification_leaves_agent_session_suspended() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    await _deliver(
        publisher,
        ledger,
        run_id,
        AgentResponse(
            answer="Which customer do you mean?",
            disposition=QuestionDisposition.NEEDS_CLARIFICATION,
        ),
    )

    assert client.stops[0]["session_status"] == "suspended"
    assert client.updates == []


async def test_non_head_progress_claim_is_deferred_without_slack_side_effects() -> None:
    ledger = DeferredProgressLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)

    assert await _open_surface(publisher, uuid4()) is None

    assert ledger.delivery.stream_state is SlackStreamState.NOT_STARTED
    assert client.starts == []
    assert client.posts == []
    assert client.api_calls == []


async def test_non_head_terminal_delivery_does_not_bypass_progress_claim() -> None:
    ledger = DeferredProgressLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)

    with pytest.raises(RunTransitionError, match="deferred behind an earlier turn"):
        await publisher.publish_safe_error(uuid4())

    assert ledger.delivery.stream_state is SlackStreamState.NOT_STARTED
    assert client.posts == []
    assert client.updates == []
    assert client.api_calls == []


async def test_answer_replay_uses_manifest_ack_without_duplicate_slack_calls() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    response = AgentResponse(answer="Grounded answer")
    await _open_surface(publisher, run_id)

    first_timestamp = await _deliver(publisher, ledger, run_id, response)
    second_timestamp = await _deliver(publisher, ledger, run_id, response)

    assert first_timestamp == second_timestamp == "2.0"
    assert len(client.starts) == 1
    assert len(client.stops) == 1


async def test_long_answer_is_ordered_and_retains_escaped_sources() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    response = AgentResponse(
        answer="Use <unsafe> & verify. " + ("x" * (MAX_STREAM_MARKDOWN + MAX_SLACK_TEXT)),
        show_sources=True,
        sources=[EvidenceReference(artifact_id="a>1", title="Runbook <ops> & support")],
    )
    await _open_surface(publisher, run_id)

    prepared = await publisher.prepare_delivery(run_id, response)
    assert await publisher.begin_delivery(run_id)
    for part_number in range(1, len(prepared.parts) + 1):
        await publisher.publish_delivery_part(run_id, prepared, part_number)
    await publisher.complete_delivery(run_id)

    assert len(prepared.parts[0]) <= MAX_STREAM_MARKDOWN
    assert all(len(part) <= MAX_SLACK_TEXT for part in prepared.parts[1:])
    assert len(client.posts) == len(prepared.parts) - 1
    assert all(post["thread_ts"] == "1.0" for post in client.posts)
    assert len({post["client_msg_id"] for post in client.posts}) == len(client.posts)
    rendered = "".join(prepared.parts)
    assert "<unsafe>" not in rendered
    assert "Runbook &lt;ops&gt; &amp; support" in rendered
    assert "`a&gt;1`" in rendered


async def test_requested_sources_do_not_duplicate_artifact_ids_in_answer_prose() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    response = AgentResponse(
        answer="The rollout was paused [art_a].",
        show_sources=True,
        sources=[EvidenceReference(artifact_id="art_a", title="Rollout notes")],
    )

    prepared = await publisher.prepare_delivery(run_id, response)
    rendered = "".join(prepared.parts)

    assert "The rollout was paused." in rendered
    assert rendered.count("art_a") == 1


async def test_ambiguous_stream_stop_retries_identical_canonical_chunks() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient(fail_first_stop=True)
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    await _deliver(
        publisher,
        ledger,
        run_id,
        AgentResponse(answer="Canonical answer"),
    )

    assert len(client.stops) == 2
    assert client.stops[0]["chunks"] == client.stops[1]["chunks"]
    assert client.stops[1]["chunks"][-1]["text"] == "Canonical answer"
    assert client.updates == []
    assert ledger.delivery.stream_state is SlackStreamState.STOPPED
    assert ledger.delivery.delivery_status is DeliveryStatus.DELIVERED


async def test_lost_stop_acknowledgement_verifies_metadata_without_duplicate_post() -> None:
    ledger = FakeLedger()
    client = LostStopAcknowledgementClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    await _deliver(
        publisher,
        ledger,
        run_id,
        AgentResponse(answer="Canonical answer"),
    )

    assert len(client.stops) == 2
    assert client.posts == []
    assert ledger.delivery.response_ts == "2.0"
    assert ledger.delivery.delivery_status is DeliveryStatus.DELIVERED


async def test_lagging_pre_stop_history_is_polled_before_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.publisher.MESSAGE_RECONCILE_POLL_SECONDS",
        0,
    )
    ledger = FakeLedger()
    client = LaggingStopReadbackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    await _deliver(
        publisher,
        ledger,
        run_id,
        AgentResponse(answer="Canonical answer"),
    )

    assert client.reply_reads == 3
    assert client.posts == []
    assert ledger.delivery.response_ts == "2.0"
    assert ledger.delivery.delivery_status is DeliveryStatus.DELIVERED


async def test_unknown_stop_readback_retries_without_duplicate_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.publisher.MESSAGE_RECONCILE_POLL_SECONDS",
        0,
    )
    ledger = FakeLedger()
    client = UnknownStopReadbackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)
    prepared = await publisher.prepare_delivery(run_id, AgentResponse(answer="Canonical answer"))
    assert await publisher.begin_delivery(run_id)

    with pytest.raises(RunTransitionError, match="could not be reconciled yet"):
        await publisher.publish_delivery_part(run_id, prepared, 1)

    assert len(client.stops) == 2
    assert client.posts == []
    assert ledger.delivery.response_ts is None


async def test_definitive_slack_rejection_is_classified_for_failure_cleanup() -> None:
    ledger = FakeLedger()
    client = DefinitiveDeliveryRejectionClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)
    prepared = await publisher.prepare_delivery(run_id, AgentResponse(answer="Canonical answer"))
    assert await publisher.begin_delivery(run_id)

    with pytest.raises(SlackDeliveryRejectedError, match="rejected canonical answer"):
        await publisher.publish_delivery_part(run_id, prepared, 1)

    assert len(client.stops) == 2
    assert client.posts == []
    assert ledger.delivery.response_ts is None


async def test_delayed_initializer_cannot_restore_processing_after_cancellation() -> None:
    ledger = FakeLedger()
    ledger.delivery = replace(
        ledger.delivery,
        stream_state=SlackStreamState.DEGRADED,
        stream_mode=SlackStreamMode.CHUNKS,
        response_ts="3.1",
        cancellation_requested=True,
        delivery_status=DeliveryStatus.CANCELLED,
    )
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()

    result = await publisher.finish_progress_surface(run_id, "2.0")

    assert result is None
    assert len(client.stops) == 1
    assert "chunks" not in client.stops[0]
    assert client.posts == []
    assert client.api_calls == []
    assert ledger.delivery.stream_state is SlackStreamState.STOPPING
    assert ledger.delivery.stream_ts == "2.0"
    assert ledger.delivery.delivery_status is DeliveryStatus.CANCELLED


async def test_silent_late_close_cannot_downgrade_concurrent_stop_finalization() -> None:
    ledger = FakeLedger()
    ledger.delivery = replace(
        ledger.delivery,
        stream_state=SlackStreamState.DEGRADED,
        stream_mode=SlackStreamMode.CHUNKS,
        cancellation_requested=True,
    )
    client = InterleavedCancellationStopClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await ledger.acknowledge_stream_open(
        run_id,
        mode=SlackStreamMode.CHUNKS,
        timestamp="2.0",
    )

    async def finish_delayed_initializer() -> None:
        assert await publisher.finish_progress_surface(run_id, "2.0") is None

    client.on_terminal_stop = finish_delayed_initializer

    await publisher.publish_cancelled(run_id)

    assert len(client.stops) == 2
    assert client.posts == []
    assert ledger.delivery.stream_state is SlackStreamState.STOPPED
    assert ledger.delivery.response_ts == "2.0"
    assert ledger.delivery.delivery_status is DeliveryStatus.CANCELLED


async def test_user_stop_race_posts_canonical_answer_instead_of_false_acknowledgement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.publisher.MESSAGE_RECONCILE_POLL_SECONDS",
        0,
    )
    ledger = FakeLedger()
    client = UserStoppedStreamClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)

    await _deliver(
        publisher,
        ledger,
        run_id,
        AgentResponse(answer="Canonical answer"),
    )

    assert len(client.stops) == 2
    assert len(client.posts) == 1
    assert client.posts[0]["text"] == "Canonical answer"
    assert ledger.delivery.response_ts == "3.1"
    assert ledger.delivery.delivery_status is DeliveryStatus.DELIVERED


async def test_user_stop_race_posts_cancellation_confirmation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "knowledge_assistant.integrations.slack.publisher.MESSAGE_RECONCILE_POLL_SECONDS",
        0,
    )
    ledger = FakeLedger()
    client = UserStoppedStreamClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)
    ledger.delivery = replace(ledger.delivery, cancellation_requested=True)

    await publisher.publish_cancelled(run_id)

    assert len(client.posts) == 1
    assert client.posts[0]["text"] == "Stopped at your request."
    assert ledger.delivery.response_ts == "3.1"
    assert ledger.delivery.delivery_status is DeliveryStatus.CANCELLED


async def test_degraded_stream_identity_is_not_accepted_as_terminal_notice() -> None:
    ledger = FakeLedger()
    ledger.delivery = replace(
        ledger.delivery,
        stream_state=SlackStreamState.DEGRADED,
        stream_mode=SlackStreamMode.CHUNKS,
        stream_ts="2.0",
    )
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)

    await publisher.publish_safe_error(uuid4())

    assert len(client.posts) == 1
    assert "couldn't complete" in client.posts[0]["text"]
    assert ledger.delivery.response_ts == "3.1"
    assert ledger.delivery.delivery_status is DeliveryStatus.FAILED


async def test_continuation_cannot_publish_before_primary_acknowledgement() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)
    prepared = await publisher.prepare_delivery(
        run_id,
        AgentResponse(answer="x" * (MAX_STREAM_MARKDOWN + 100)),
    )
    assert len(prepared.parts) > 1
    assert await publisher.begin_delivery(run_id)

    with pytest.raises(RunTransitionError, match="cannot precede"):
        await publisher.publish_delivery_part(run_id, prepared, 2)

    assert client.posts == []


async def test_partial_delivery_preserves_answer_and_appends_incomplete_notice() -> None:
    ledger = FakeLedger()
    client = FakeSlackClient()
    publisher = _publisher(client, ledger)
    run_id = uuid4()
    await _open_surface(publisher, run_id)
    prepared = await publisher.prepare_delivery(
        run_id,
        AgentResponse(answer="x" * (MAX_STREAM_MARKDOWN + 100)),
    )
    assert await publisher.begin_delivery(run_id)
    await publisher.publish_delivery_part(run_id, prepared, 1)

    await publisher.publish_incomplete_delivery_notice(run_id)

    assert len(client.stops) == 1
    assert len(client.posts) == 1
    assert "couldn't finish posting" in client.posts[0]["text"]
    assert ledger.delivery.delivery_status is DeliveryStatus.FAILED


async def test_degraded_final_post_uses_stable_uuid5_client_message_id() -> None:
    run_id = uuid4()
    first_client = FailingStartClient()
    second_client = FailingStartClient()
    first_ledger = FakeLedger()
    second_ledger = FakeLedger()
    first_publisher = _publisher(first_client, first_ledger)
    second_publisher = _publisher(second_client, second_ledger)

    await _open_surface(first_publisher, run_id)
    await _open_surface(second_publisher, run_id)
    await _deliver(
        first_publisher,
        first_ledger,
        run_id,
        AgentResponse(answer="Canonical answer"),
    )
    await _deliver(
        second_publisher,
        second_ledger,
        run_id,
        AgentResponse(answer="Canonical answer"),
    )

    first_id = UUID(first_client.posts[0]["client_msg_id"])
    second_id = UUID(second_client.posts[0]["client_msg_id"])
    assert first_id.version == 5
    assert first_id == second_id
