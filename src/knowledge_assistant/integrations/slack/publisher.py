"""Retry-safe native Slack progress streams and final-answer delivery."""

from __future__ import annotations

import asyncio
import hashlib
import re
import time
import uuid
from contextlib import suppress
from enum import StrEnum
from typing import Literal, Never

import structlog
from pydantic import BaseModel, ConfigDict, Field, model_validator
from slack_sdk.errors import SlackApiError
from slack_sdk.web.async_client import AsyncWebClient

from knowledge_assistant.agent.citations import hide_artifact_citations
from knowledge_assistant.agent.models import AgentResponse, ProgressEvent, ProgressStage
from knowledge_assistant.persistence.models import (
    DeliveryStatus,
    SlackStreamMode,
    SlackStreamState,
)
from knowledge_assistant.persistence.repositories import (
    DeliveryManifest,
    DeliveryState,
    RunLedger,
    RunTransitionError,
)

logger = structlog.get_logger(__name__)

MAX_STREAM_MARKDOWN = 12_000
MAX_SLACK_TEXT = 4_000
DELIVERY_MANIFEST_VERSION = 1
STREAM_OPEN_WAIT_SECONDS = 25.0
STREAM_OPEN_POLL_SECONDS = 0.25
MESSAGE_RECONCILE_ATTEMPTS = 3
MESSAGE_RECONCILE_POLL_SECONDS = 0.25
_SOURCES_HEADER = "\n\n**Sources**\n\n"
_CONTINUATION_HEADER = "**Continued**\n\n"
_INITIAL_TASK_KEY = "understanding"
_DELIVERY_METADATA_EVENT_TYPE = "grounded_qa_delivery"
_SAFE_ERROR_TEXT = "I couldn't complete that request right now. Please try again shortly."
_CANCELLED_TEXT = "Stopped at your request."
_INCOMPLETE_DELIVERY_TEXT = (
    "I couldn't finish posting the full response. Please retry the question for a complete answer."
)
_RETRYABLE_SLACK_DELIVERY_ERRORS = frozenset(
    {
        "fatal_error",
        "internal_error",
        "ratelimited",
        "request_timeout",
        "service_unavailable",
    }
)

_PROGRESS_TITLES = {
    ProgressStage.THINKING: "Understanding the question",
    ProgressStage.SEARCHING: "Searching company knowledge",
    ProgressStage.REVIEWING: "Reviewing supporting evidence",
    ProgressStage.DRAFTING: "Drafting a grounded answer",
    ProgressStage.VERIFYING: "Verifying claims and sources",
    ProgressStage.TIGHTENING: "Tightening the final response",
}
_PROGRESS_TITLES_BY_SEQUENCE = {
    10: _PROGRESS_TITLES[ProgressStage.THINKING],
    20: _PROGRESS_TITLES[ProgressStage.SEARCHING],
    30: _PROGRESS_TITLES[ProgressStage.REVIEWING],
    40: _PROGRESS_TITLES[ProgressStage.SEARCHING],
    50: _PROGRESS_TITLES[ProgressStage.REVIEWING],
    60: _PROGRESS_TITLES[ProgressStage.DRAFTING],
    70: _PROGRESS_TITLES[ProgressStage.VERIFYING],
    80: _PROGRESS_TITLES[ProgressStage.TIGHTENING],
    90: _PROGRESS_TITLES[ProgressStage.VERIFYING],
}


class PreparedDelivery(BaseModel):
    """Deterministic message parts returned through an Inngest step boundary."""

    model_config = ConfigDict(frozen=True)

    version: int = Field(default=DELIVERY_MANIFEST_VERSION, ge=1)
    parts: tuple[str, ...] = Field(min_length=1, max_length=100)
    content_hashes: tuple[str, ...] = Field(min_length=1, max_length=100)
    completion_title: str = Field(default="Answer ready", min_length=1, max_length=256)
    completion_task_key: str = Field(
        default=_INITIAL_TASK_KEY,
        min_length=1,
        max_length=128,
    )
    session_status: Literal["active", "suspended"] = "active"

    @model_validator(mode="after")
    def validate_part_hashes(self) -> PreparedDelivery:
        if len(self.parts) != len(self.content_hashes):
            raise ValueError("Delivery parts and hashes must have the same length")
        for part, content_hash in zip(self.parts, self.content_hashes, strict=True):
            if _content_hash(part) != content_hash:
                raise ValueError("Delivery part content does not match its hash")
        return self


class ProgressSurfaceAction(StrEnum):
    """Durable orchestration action selected without making a Slack API call."""

    START = "start"
    READY = "ready"
    WAIT = "wait"
    DEGRADED = "degraded"


class DeliveryVerification(StrEnum):
    """Read-back result after Slack says a stream is already terminal."""

    VERIFIED = "verified"
    CONFIRMED_ABSENT = "confirmed_absent"
    UNKNOWN = "unknown"


class DeliveryReconciliationPendingError(RunTransitionError):
    """Raised when Slack delivery may have committed but cannot yet be proven."""


class SlackDeliveryRejectedError(RuntimeError):
    """Raised when Slack definitively rejects canonical answer content."""


class ProgressSurfaceClaim(BaseModel):
    """Serializable claim passed between independent Inngest steps."""

    model_config = ConfigDict(frozen=True)

    action: ProgressSurfaceAction
    timestamp: str | None = None

    @model_validator(mode="after")
    def validate_timestamp(self) -> ProgressSurfaceClaim:
        if self.action is ProgressSurfaceAction.READY and self.timestamp is None:
            raise ValueError("A ready progress surface requires a timestamp")
        if self.action is not ProgressSurfaceAction.READY and self.timestamp is not None:
            raise ValueError("Only a ready progress surface can carry a timestamp")
        return self


def _escape_slack_text(text: str) -> str:
    """Prevent generated text from becoming Slack mention/link control syntax."""

    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _escape_source_label(text: str) -> str:
    escaped = _escape_slack_text(" ".join(text.split()))
    return re.sub(r"([\\`*_{}\[\]()#+.!|~-])", r"\\\1", escaped)


def _safe_inline_code(text: str) -> str:
    return _escape_slack_text(" ".join(text.replace("`", "'").split()))


def _client_message_id(run_id: uuid.UUID, delivery_kind: str) -> str:
    # Slack can deduplicate an ambiguous postMessage retry when every attempt reuses this ID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"slack-qa-agent:{run_id}:{delivery_kind}"))


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _delivery_metadata(
    run_id: uuid.UUID,
    *,
    delivery_kind: str,
    content_hash: str,
) -> dict[str, object]:
    """Attach a canonical identity that can be verified after an ambiguous Slack write."""

    return {
        "event_type": _DELIVERY_METADATA_EVENT_TYPE,
        "event_payload": {
            "run_id": str(run_id),
            "delivery_kind": delivery_kind,
            "content_hash": content_hash,
        },
    }


def _message_has_metadata(
    message: object,
    *,
    timestamp: str,
    expected_metadata: dict[str, object],
) -> bool:
    if not isinstance(message, dict) or str(message.get("ts", "")) != timestamp:
        return False
    metadata = message.get("metadata")
    expected_payload = expected_metadata.get("event_payload")
    if not isinstance(metadata, dict) or not isinstance(expected_payload, dict):
        return False
    payload = metadata.get("event_payload")
    return (
        metadata.get("event_type") == expected_metadata.get("event_type")
        and isinstance(payload, dict)
        and all(payload.get(key) == value for key, value in expected_payload.items())
    )


def _split_text(text: str, limit: int) -> list[str]:
    """Split text at readable boundaries without discarding answer content."""

    chunks: list[str] = []
    remaining = text
    while len(remaining) > limit:
        split_at = limit
        separator_length = 0
        for separator in ("\n\n", "\n", " "):
            candidate = remaining.rfind(separator, 0, limit + 1)
            if candidate >= limit // 2:
                split_at = candidate
                separator_length = len(separator)
                break

        chunks.append(remaining[:split_at].rstrip())
        remaining = remaining[split_at + separator_length :].lstrip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _format_answer_parts(response: AgentResponse) -> tuple[str, ...]:
    # Artifact markers are an internal grounding contract. Slack renders the structured source
    # list only when requested, avoiding duplicate IDs in both prose and the source section.
    display_answer = hide_artifact_citations(response.answer)
    escaped_answer = _escape_slack_text(display_answer)
    source_lines = [
        f"- {_escape_source_label(source.title)} (`{_safe_inline_code(source.artifact_id)}`)"
        for source in response.sources
    ]
    sources_section = (
        _SOURCES_HEADER + "\n".join(source_lines) if response.show_sources and source_lines else ""
    )
    full_text = escaped_answer + sources_section
    if len(full_text) <= MAX_STREAM_MARKDOWN:
        return (full_text,)

    primary, remainder = _take_readable_prefix(full_text, MAX_STREAM_MARKDOWN)
    continuation_limit = MAX_SLACK_TEXT - len(_CONTINUATION_HEADER)
    continuations = _split_text(remainder, continuation_limit)
    return (
        primary,
        *(_CONTINUATION_HEADER + chunk for chunk in continuations),
    )


def _take_readable_prefix(text: str, limit: int) -> tuple[str, str]:
    if len(text) <= limit:
        return text, ""
    split_at = limit
    separator_length = 0
    for separator in ("\n\n", "\n", " "):
        candidate = text.rfind(separator, 0, limit + 1)
        if candidate >= limit // 2:
            split_at = candidate
            separator_length = len(separator)
            break
    return (
        text[:split_at].rstrip(),
        text[split_at + separator_length :].lstrip(),
    )


def _markdown_blocks(text: str) -> list[dict[str, str]]:
    return [{"type": "markdown", "text": text}]


def _work_task_id(run_id: uuid.UUID, task_key: str) -> str:
    return f"{run_id}:{task_key}"


def _progress_task_key(sequence: int) -> str:
    return f"stage-{sequence}"


def _progress_title_for_sequence(sequence: int) -> str:
    return _PROGRESS_TITLES_BY_SEQUENCE.get(sequence, "Completed step")


def _task_update(
    run_id: uuid.UUID,
    *,
    title: str,
    status: str,
    task_key: str,
    hide_title: bool = False,
) -> dict[str, object]:
    return {
        "type": "task_update",
        "id": _work_task_id(run_id, task_key),
        "title": title,
        "hide_title": hide_title,
        "status": status,
    }


def _completion_plan_title(stream_ts: str | None) -> str:
    """Create one replay-safe elapsed-time label for the completed stream."""

    if stream_ts is None:
        return "Answer ready"
    try:
        elapsed_seconds = max(0, int(time.time() - float(stream_ts)))
    except ValueError:
        return "Answer ready"
    if elapsed_seconds < 60:
        return f"Answered in {elapsed_seconds}s"
    minutes, seconds = divmod(elapsed_seconds, 60)
    return f"Answered in {minutes}m {seconds}s"


def _manifest_part(manifest: DeliveryManifest, part_number: int) -> tuple[str, str | None]:
    try:
        part = manifest.parts[part_number - 1]
    except IndexError as exc:
        raise RunTransitionError(f"Delivery part does not exist: {part_number}") from exc
    if part.part_number != part_number:
        raise RunTransitionError("Delivery manifest parts are not ordered")
    return part.content_hash, part.slack_message_ts


def _is_already_stopped_error(exc: SlackApiError) -> bool:
    return str(exc.response.get("error", "")) == "message_not_in_streaming_state"


def _raise_classified_delivery_error(exc: SlackApiError) -> Never:
    """Separate definitive payload rejection from retryable Slack availability failures."""

    status_code = int(getattr(exc.response, "status_code", 0) or 0)
    error_code = str(exc.response.get("error", ""))
    if status_code == 429 or status_code >= 500 or error_code in _RETRYABLE_SLACK_DELIVERY_ERRORS:
        raise DeliveryReconciliationPendingError(
            "Slack final delivery is temporarily unavailable"
        ) from exc
    raise SlackDeliveryRejectedError("Slack rejected canonical answer delivery") from exc


class SlackPublisher:
    """Own native Slack stream state and canonical, ordered answer delivery."""

    def __init__(
        self,
        client: AsyncWebClient,
        ledger: RunLedger,
    ) -> None:
        self._client = client
        self._ledger = ledger

    async def claim_progress_surface(self, run_id: uuid.UUID) -> ProgressSurfaceClaim:
        """Claim native progress using only durable database operations.

        Slack stream creation is deliberately a separate Inngest step. A returned Slack
        timestamp is therefore durable workflow output before a later step acknowledges it in
        PostgreSQL, so a database retry never calls ``chat.startStream`` again.
        """

        delivery = await self._ledger.get_delivery(run_id)
        if delivery.cancellation_requested:
            raise RunTransitionError(
                f"Agent run cannot open progress after cancellation was requested: {run_id}"
            )
        existing = self._existing_progress_claim(run_id, delivery)
        if existing is not None:
            return existing

        claimed = await self._ledger.transition_stream(
            run_id,
            expected_state=SlackStreamState.NOT_STARTED,
            target_state=SlackStreamState.OPENING,
            mode=SlackStreamMode.CHUNKS,
        )
        if claimed:
            return ProgressSurfaceClaim(action=ProgressSurfaceAction.START)

        delivery = await self._ledger.get_delivery(run_id)
        existing = self._existing_progress_claim(run_id, delivery)
        if existing is not None:
            return existing
        # The partial unique index deferred this run behind the active turn for its conversation.
        return ProgressSurfaceClaim(action=ProgressSurfaceAction.WAIT)

    def _existing_progress_claim(
        self,
        run_id: uuid.UUID,
        delivery: DeliveryState,
    ) -> ProgressSurfaceClaim | None:
        if delivery.response_ts is not None:
            return ProgressSurfaceClaim(
                action=ProgressSurfaceAction.READY,
                timestamp=delivery.response_ts,
            )
        if delivery.stream_state in {
            SlackStreamState.OPEN,
            SlackStreamState.STOPPING,
            SlackStreamState.STOPPED,
        } or (
            delivery.stream_state is SlackStreamState.UNCERTAIN and delivery.stream_ts is not None
        ):
            if delivery.stream_ts is None:
                raise RunTransitionError(f"Persisted Slack stream has no timestamp: {run_id}")
            return ProgressSurfaceClaim(
                action=ProgressSurfaceAction.READY,
                timestamp=delivery.stream_ts,
            )
        if delivery.stream_state is SlackStreamState.OPENING:
            # Another durable step owns the non-idempotent remote start. Waiting is safer than
            # creating a second message if that step's response is still in flight.
            return ProgressSurfaceClaim(action=ProgressSurfaceAction.WAIT)
        if delivery.stream_state in {
            SlackStreamState.DEGRADED,
            SlackStreamState.UNCERTAIN,
        }:
            return ProgressSurfaceClaim(action=ProgressSurfaceAction.DEGRADED)
        if delivery.stream_state is SlackStreamState.NOT_STARTED:
            return None
        raise RunTransitionError(
            f"Cannot claim progress from {delivery.stream_state.value}: {run_id}"
        )

    async def start_claimed_stream(self, run_id: uuid.UUID) -> str | None:
        """Perform the one non-idempotent Slack start call and return its remote identity."""

        delivery = await self._ledger.get_delivery(run_id)
        if delivery.cancellation_requested or delivery.stream_state is not SlackStreamState.OPENING:
            return None
        try:
            result = await self._client.chat_startStream(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                recipient_team_id=delivery.team_id,
                recipient_user_id=delivery.user_id,
                task_display_mode="plan",
                chunks=[
                    {"type": "plan_update", "title": "Working on your request"},
                    _task_update(
                        run_id,
                        title="Understanding the request",
                        status="in_progress",
                        task_key=_INITIAL_TASK_KEY,
                    ),
                ],
            )
            timestamp = str(result.get("ts", ""))
            if not timestamp:
                raise RuntimeError("Slack stream response did not include a timestamp")
            return timestamp
        except Exception as exc:
            # The start method has no idempotency key. Treat an ambiguous response as terminal
            # degradation so Inngest does not retry it and create duplicate progress messages.
            logger.warning(
                "slack_stream_open_failed",
                agent_run_id=str(run_id),
                exception_class=type(exc).__name__,
            )
            return None

    async def finish_progress_surface(
        self,
        run_id: uuid.UUID,
        timestamp: str | None,
    ) -> str | None:
        """Acknowledge a persisted Slack step result and reconcile a concurrent Stop."""

        if timestamp is None:
            delivery = await self._ledger.get_delivery(run_id)
            if (
                delivery.cancellation_requested
                or delivery.response_ts is not None
                or delivery.delivery_status is not DeliveryStatus.PENDING
            ):
                # A delayed initializer must never restore processing after cancellation or
                # terminal delivery has already won.
                return None
            return await self._degrade_progress_surface(run_id)
        acknowledgement = await self._ledger.acknowledge_stream_open(
            run_id,
            mode=SlackStreamMode.CHUNKS,
            timestamp=timestamp,
        )
        if acknowledgement.should_close:
            # Cancellation cleanup is the sole owner of the user-visible Stop notice. The
            # delayed initializer only closes the late progress surface; otherwise cleanup
            # could post a fallback while this path finalizes the stream with the same text.
            await self._close_redundant_open_stream(
                run_id,
                timestamp,
                should_restore_processing=not acknowledgement.cancellation_requested,
            )
            return None
        return timestamp

    async def _wait_for_stream_open(self, run_id: uuid.UUID) -> DeliveryState:
        """Let the other idempotent worker finish its bounded startStream call."""

        loop = asyncio.get_running_loop()
        deadline = loop.time() + STREAM_OPEN_WAIT_SECONDS
        delivery = await self._ledger.get_delivery(run_id)
        while delivery.stream_state == SlackStreamState.OPENING and loop.time() < deadline:
            await asyncio.sleep(STREAM_OPEN_POLL_SECONDS)
            delivery = await self._ledger.get_delivery(run_id)
        return delivery

    async def publish_progress(self, run_id: uuid.UUID, event: ProgressEvent) -> bool:
        """Publish a code-owned stage without exposing model reasoning or retrieved content."""

        delivery = await self._ledger.get_delivery(run_id)
        if delivery.cancellation_requested or delivery.stream_state != SlackStreamState.OPEN:
            return False
        try:
            should_publish = await self._ledger.advance_progress(run_id, event.sequence)
        except RunTransitionError:
            # Finalization may have moved the stream after the read above.
            return False
        if not should_publish:
            return False

        title = _PROGRESS_TITLES[event.stage]
        previous_sequence = delivery.last_progress_sequence
        current_task_key = _progress_task_key(event.sequence)
        chunks: list[dict[str, object]] = []
        if previous_sequence == 0:
            chunks.append(
                _task_update(
                    run_id,
                    title="Understanding the request",
                    status="complete",
                    task_key=_INITIAL_TASK_KEY,
                )
            )
        else:
            chunks.append(
                _task_update(
                    run_id,
                    title=_progress_title_for_sequence(previous_sequence),
                    status="complete",
                    task_key=_progress_task_key(previous_sequence),
                )
            )
        chunks.append(
            _task_update(
                run_id,
                title=title,
                status="in_progress",
                task_key=current_task_key,
            )
        )
        try:
            if delivery.stream_ts is None:
                raise RunTransitionError(f"Open Slack stream has no timestamp: {run_id}")
            await self._client.chat_appendStream(
                channel=delivery.channel_id,
                ts=delivery.stream_ts,
                # Slack shows the active plan task before completed tasks. Each stage gets a
                # stable ID, so the UI retains a readable history while the newest work stays
                # at the top.
                chunks=chunks,
            )
        except Exception as exc:
            # Progress is informative and intentionally best-effort. Final delivery remains strict.
            logger.warning(
                "slack_progress_publish_failed",
                agent_run_id=str(run_id),
                progress_sequence=event.sequence,
                progress_stage=event.stage.value,
                exception_class=type(exc).__name__,
            )
            return False
        return True

    async def prepare_delivery(
        self,
        run_id: uuid.UUID,
        response: AgentResponse,
    ) -> PreparedDelivery:
        parts = _format_answer_parts(response)
        content_hashes = tuple(_content_hash(part) for part in parts)
        delivery = await self._ledger.get_delivery(run_id)
        prepared = PreparedDelivery(
            parts=parts,
            content_hashes=content_hashes,
            completion_title=_completion_plan_title(delivery.stream_ts),
            completion_task_key=(
                _INITIAL_TASK_KEY
                if delivery.last_progress_sequence == 0
                else _progress_task_key(delivery.last_progress_sequence)
            ),
            session_status="suspended" if response.requires_user_input else "active",
        )
        await self._ledger.install_delivery_manifest(
            run_id,
            version=prepared.version,
            part_hashes=prepared.content_hashes,
        )
        return prepared

    async def begin_delivery(self, run_id: uuid.UUID) -> bool:
        return await self._ledger.claim_delivery(run_id)

    async def publish_delivery_part(
        self,
        run_id: uuid.UUID,
        prepared: PreparedDelivery,
        part_number: int,
    ) -> str:
        """Deliver one immutable part; persisted acknowledgement is the replay authority."""

        if not 1 <= part_number <= len(prepared.parts):
            raise ValueError(f"Invalid delivery part number: {part_number}")
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.cancellation_requested:
            raise RunTransitionError(
                f"Agent run cannot publish after cancellation was requested: {run_id}"
            )
        manifest = await self._ledger.get_delivery_manifest(run_id)
        if manifest is None:
            raise RunTransitionError(f"Agent run has no delivery manifest: {run_id}")
        if manifest.version != prepared.version:
            raise RunTransitionError(f"Delivery manifest version changed: {run_id}")
        if any(part.acknowledged_at is None for part in manifest.parts[: part_number - 1]):
            raise RunTransitionError(
                f"Delivery part {part_number} cannot precede an earlier part: {run_id}"
            )

        expected_hash, acknowledged_ts = _manifest_part(manifest, part_number)
        content_hash = prepared.content_hashes[part_number - 1]
        if expected_hash != content_hash:
            raise RunTransitionError(f"Delivery part content changed: {run_id}:{part_number}")
        if acknowledged_ts is not None:
            return acknowledged_ts

        text = prepared.parts[part_number - 1]
        if part_number == 1:
            return await self._publish_primary_part(
                run_id,
                text,
                content_hash,
                completion_title=prepared.completion_title,
                completion_task_key=prepared.completion_task_key,
                session_status=prepared.session_status,
            )
        return await self._publish_continuation_part(
            run_id,
            part_number=part_number,
            text=text,
            content_hash=content_hash,
        )

    async def complete_delivery(self, run_id: uuid.UUID) -> None:
        await self._ledger.mark_delivery_delivered(run_id)

    async def publish_safe_error(self, run_id: uuid.UUID) -> None:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.delivery_status in {DeliveryStatus.DELIVERED, DeliveryStatus.CANCELLED}:
            await self._set_session_status_best_effort(delivery, "active", run_id=run_id)
            return
        await self._publish_terminal_notice(run_id, _SAFE_ERROR_TEXT, is_error=True)
        await self._ledger.mark_delivery_failed(run_id)

    async def publish_cancelled(self, run_id: uuid.UUID) -> None:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.delivery_status in {DeliveryStatus.DELIVERED, DeliveryStatus.CANCELLED}:
            await self._set_session_status_best_effort(delivery, "active", run_id=run_id)
            return
        await self._publish_terminal_notice(run_id, _CANCELLED_TEXT, is_error=True)
        await self._ledger.mark_delivery_cancelled(run_id)

    async def publish_incomplete_delivery_notice(self, run_id: uuid.UUID) -> None:
        """Preserve acknowledged answer parts and append a retry-safe incompleteness warning."""

        delivery = await self._ledger.get_delivery(run_id)
        await self._client.chat_postMessage(
            channel=delivery.channel_id,
            thread_ts=delivery.thread_ts,
            text=_INCOMPLETE_DELIVERY_TEXT,
            blocks=_markdown_blocks(_INCOMPLETE_DELIVERY_TEXT),
            client_msg_id=_client_message_id(run_id, "delivery-incomplete"),
        )
        await self._set_session_status_strict(delivery, "active", run_id=run_id)
        await self._ledger.mark_delivery_failed(run_id)

    async def abandon_unreconciled_cancellation(self, run_id: uuid.UUID) -> None:
        """Release a cancelled run after remote terminal identity stays unknowable."""

        delivery = await self._degrade_unreconciled_surface(run_id)
        await self._set_session_status_best_effort(delivery, "active", run_id=run_id)
        await self._ledger.mark_delivery_cancelled(run_id)

    async def abandon_unreconciled_failure(self, run_id: uuid.UUID) -> None:
        """Release a failed run without risking another ambiguous user-visible write."""

        delivery = await self._degrade_unreconciled_surface(run_id)
        await self._set_session_status_best_effort(delivery, "active", run_id=run_id)
        await self._ledger.mark_delivery_failed(run_id)

    async def _degrade_unreconciled_surface(self, run_id: uuid.UUID) -> DeliveryState:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.stream_state in {
            SlackStreamState.OPENING,
            SlackStreamState.OPEN,
            SlackStreamState.STOPPING,
            SlackStreamState.UNCERTAIN,
        }:
            try:
                await self._ledger.transition_stream(
                    run_id,
                    expected_state=delivery.stream_state,
                    target_state=SlackStreamState.DEGRADED,
                )
            except RunTransitionError:
                latest_delivery = await self._ledger.get_delivery(run_id)
                if latest_delivery.stream_state not in {
                    SlackStreamState.NOT_STARTED,
                    SlackStreamState.STOPPED,
                    SlackStreamState.DEGRADED,
                }:
                    raise
        return await self._ledger.get_delivery(run_id)

    async def _close_redundant_open_stream(
        self,
        run_id: uuid.UUID,
        timestamp: str,
        *,
        should_restore_processing: bool,
    ) -> None:
        """Close a late native start after another worker abandoned progress rendering."""

        delivery = await self._ledger.get_delivery(run_id)
        try:
            await self._client.chat_stopStream(
                channel=delivery.channel_id,
                ts=timestamp,
            )
        except SlackApiError as exc:
            if not _is_already_stopped_error(exc):
                raise
        if should_restore_processing:
            await self._ledger.transition_stream(
                run_id,
                expected_state=SlackStreamState.STOPPING,
                target_state=SlackStreamState.DEGRADED,
            )
            latest_delivery = await self._ledger.get_delivery(run_id)
            if (
                not latest_delivery.cancellation_requested
                and latest_delivery.response_ts is None
                and latest_delivery.delivery_status is DeliveryStatus.PENDING
            ):
                await self._set_session_status_strict(
                    latest_delivery,
                    "processing",
                    run_id=run_id,
                )
        return None

    async def _degrade_progress_surface(self, run_id: uuid.UUID) -> str | None:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.stream_state == SlackStreamState.OPENING:
            await self._ledger.transition_stream(
                run_id,
                expected_state=SlackStreamState.OPENING,
                target_state=SlackStreamState.UNCERTAIN,
            )
            delivery = await self._ledger.get_delivery(run_id)
        if delivery.stream_state in {
            SlackStreamState.NOT_STARTED,
            SlackStreamState.UNCERTAIN,
            SlackStreamState.OPEN,
            SlackStreamState.STOPPING,
        }:
            claimed = await self._ledger.transition_stream(
                run_id,
                expected_state=delivery.stream_state,
                target_state=SlackStreamState.DEGRADED,
            )
            if not claimed and delivery.stream_state == SlackStreamState.NOT_STARTED:
                delivery = await self._ledger.get_delivery(run_id)
                if delivery.stream_state == SlackStreamState.NOT_STARTED:
                    raise RunTransitionError(
                        f"Slack progress is deferred behind an earlier turn: {run_id}"
                    )
        delivery = await self._ledger.get_delivery(run_id)
        if (
            not delivery.cancellation_requested
            and delivery.response_ts is None
            and delivery.delivery_status is DeliveryStatus.PENDING
        ):
            await self._set_session_status_best_effort(delivery, "processing", run_id=run_id)
        return None

    async def _stream_contains_delivery(
        self,
        run_id: uuid.UUID,
        delivery: DeliveryState,
        *,
        timestamp: str,
        metadata: dict[str, object],
    ) -> DeliveryVerification:
        """Verify a canonical final write after Slack reports an already-stopped stream."""

        metadata_absence_observations = 0
        for attempt in range(MESSAGE_RECONCILE_ATTEMPTS):
            try:
                result = await self._client.conversations_replies(
                    channel=delivery.channel_id,
                    ts=delivery.thread_ts,
                    oldest=timestamp,
                    latest=timestamp,
                    inclusive=True,
                    limit=100,
                )
                messages: object = result.get("messages", [])
                if isinstance(messages, list):
                    matching_message = next(
                        (
                            message
                            for message in messages
                            if isinstance(message, dict) and str(message.get("ts", "")) == timestamp
                        ),
                        None,
                    )
                    if matching_message is not None:
                        if _message_has_metadata(
                            matching_message,
                            timestamp=timestamp,
                            expected_metadata=metadata,
                        ):
                            return DeliveryVerification.VERIFIED
                        # Slack history can briefly expose the pre-stop form of this same
                        # streaming message. Require every bounded observation to agree before
                        # choosing an immutable fallback; any mixed result remains unknown.
                        metadata_absence_observations += 1
            except Exception as exc:
                logger.warning(
                    "slack_stream_reconciliation_read_failed",
                    agent_run_id=str(run_id),
                    attempt=attempt + 1,
                    exception_class=type(exc).__name__,
                )
            if attempt + 1 < MESSAGE_RECONCILE_ATTEMPTS:
                await asyncio.sleep(MESSAGE_RECONCILE_POLL_SECONDS)
        if metadata_absence_observations == MESSAGE_RECONCILE_ATTEMPTS:
            return DeliveryVerification.CONFIRMED_ABSENT
        return DeliveryVerification.UNKNOWN

    async def _publish_primary_part(
        self,
        run_id: uuid.UUID,
        text: str,
        content_hash: str,
        *,
        completion_title: str,
        completion_task_key: str,
        session_status: Literal["active", "suspended"],
    ) -> str:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.stream_state == SlackStreamState.OPEN:
            await self._ledger.transition_stream(
                run_id,
                expected_state=SlackStreamState.OPEN,
                target_state=SlackStreamState.STOPPING,
            )
            delivery = await self._ledger.get_delivery(run_id)
            if delivery.stream_ts is None:
                raise RunTransitionError(f"Stopping Slack stream has no timestamp: {run_id}")
            metadata = _delivery_metadata(
                run_id,
                delivery_kind="answer:1",
                content_hash=content_hash,
            )
            try:
                await self._client.chat_stopStream(
                    channel=delivery.channel_id,
                    ts=delivery.stream_ts,
                    chunks=[
                        _task_update(
                            run_id,
                            title="Answer ready",
                            status="complete",
                            task_key=completion_task_key,
                            hide_title=True,
                        ),
                        {
                            "type": "plan_update",
                            "title": completion_title,
                        },
                        {"type": "markdown_text", "text": text},
                    ],
                    metadata=metadata,
                    session_status=session_status,
                )
                await self._ledger.transition_stream(
                    run_id,
                    expected_state=SlackStreamState.STOPPING,
                    target_state=SlackStreamState.STOPPED,
                )
                return await self._acknowledge_primary(
                    run_id,
                    timestamp=delivery.stream_ts,
                    content_hash=content_hash,
                )
            except Exception as exc:
                logger.warning(
                    "slack_stream_stop_uncertain",
                    agent_run_id=str(run_id),
                    exception_class=type(exc).__name__,
                )
                with suppress(RunTransitionError):
                    await self._ledger.transition_stream(
                        run_id,
                        expected_state=SlackStreamState.STOPPING,
                        target_state=SlackStreamState.UNCERTAIN,
                    )
                return await self._reconcile_primary(
                    run_id,
                    text,
                    content_hash,
                    completion_title=completion_title,
                    completion_task_key=completion_task_key,
                    session_status=session_status,
                )

        if delivery.stream_state in {
            SlackStreamState.STOPPING,
            SlackStreamState.UNCERTAIN,
            SlackStreamState.STOPPED,
        }:
            return await self._reconcile_primary(
                run_id,
                text,
                content_hash,
                completion_title=completion_title,
                completion_task_key=completion_task_key,
                session_status=session_status,
            )
        if delivery.stream_state != SlackStreamState.DEGRADED:
            await self._degrade_progress_surface(run_id)
            delivery = await self._ledger.get_delivery(run_id)
            if delivery.stream_state == SlackStreamState.NOT_STARTED:
                raise RunTransitionError(
                    f"Slack answer delivery is deferred behind an earlier turn: {run_id}"
                )
        delivery = await self._ledger.get_delivery(run_id)
        return await self._post_primary_fallback(
            run_id,
            delivery,
            text=text,
            content_hash=content_hash,
            session_status=session_status,
        )

    async def _reconcile_primary(
        self,
        run_id: uuid.UUID,
        text: str,
        content_hash: str,
        *,
        completion_title: str,
        completion_task_key: str,
        session_status: Literal["active", "suspended"],
    ) -> str:
        """Retry the same canonical stop without rewriting a finalized Slack message."""

        delivery = await self._ledger.get_delivery(run_id)
        if delivery.stream_ts is None:
            raise RunTransitionError(f"Uncertain Slack stream has no timestamp: {run_id}")
        metadata = _delivery_metadata(
            run_id,
            delivery_kind="answer:1",
            content_hash=content_hash,
        )
        was_delivered_on_stream = False
        try:
            # The prior ambiguous stop may already have completed successfully.
            await self._client.chat_stopStream(
                channel=delivery.channel_id,
                ts=delivery.stream_ts,
                chunks=[
                    _task_update(
                        run_id,
                        title="Answer ready",
                        status="complete",
                        task_key=completion_task_key,
                        hide_title=True,
                    ),
                    {"type": "plan_update", "title": completion_title},
                    {"type": "markdown_text", "text": text},
                ],
                metadata=metadata,
                session_status=session_status,
            )
            was_delivered_on_stream = True
        except SlackApiError as exc:
            if not _is_already_stopped_error(exc):
                _raise_classified_delivery_error(exc)
            verification = await self._stream_contains_delivery(
                run_id,
                delivery,
                timestamp=delivery.stream_ts,
                metadata=metadata,
            )
            if verification is DeliveryVerification.UNKNOWN:
                raise DeliveryReconciliationPendingError(
                    f"Slack final delivery could not be reconciled yet: {run_id}"
                ) from exc
            was_delivered_on_stream = verification is DeliveryVerification.VERIFIED
        if delivery.stream_state in {SlackStreamState.STOPPING, SlackStreamState.UNCERTAIN}:
            await self._ledger.transition_stream(
                run_id,
                expected_state=delivery.stream_state,
                target_state=SlackStreamState.STOPPED,
            )
        if was_delivered_on_stream:
            await self._set_session_status_strict(delivery, session_status, run_id=run_id)
            return await self._acknowledge_primary(
                run_id,
                timestamp=delivery.stream_ts,
                content_hash=content_hash,
            )
        delivery = await self._ledger.get_delivery(run_id)
        return await self._post_primary_fallback(
            run_id,
            delivery,
            text=text,
            content_hash=content_hash,
            session_status=session_status,
        )

    async def _post_primary_fallback(
        self,
        run_id: uuid.UUID,
        delivery: DeliveryState,
        *,
        text: str,
        content_hash: str,
        session_status: Literal["active", "suspended"],
    ) -> str:
        try:
            result = await self._client.chat_postMessage(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                text=text,
                blocks=_markdown_blocks(text),
                metadata=_delivery_metadata(
                    run_id,
                    delivery_kind="answer:1",
                    content_hash=content_hash,
                ),
                client_msg_id=_client_message_id(run_id, "answer:1"),
            )
        except SlackApiError as exc:
            _raise_classified_delivery_error(exc)
        timestamp = str(result.get("ts", ""))
        if not timestamp:
            raise RuntimeError("Slack answer response did not include a timestamp")
        await self._set_session_status_strict(delivery, session_status, run_id=run_id)
        return await self._acknowledge_primary(
            run_id,
            timestamp=timestamp,
            content_hash=content_hash,
        )

    async def _acknowledge_primary(
        self,
        run_id: uuid.UUID,
        *,
        timestamp: str,
        content_hash: str,
    ) -> str:
        await self._ledger.set_response(run_id, timestamp)
        await self._ledger.acknowledge_delivery_part(
            run_id,
            part_number=1,
            content_hash=content_hash,
            slack_message_ts=timestamp,
        )
        return timestamp

    async def _publish_continuation_part(
        self,
        run_id: uuid.UUID,
        *,
        part_number: int,
        text: str,
        content_hash: str,
    ) -> str:
        delivery = await self._ledger.get_delivery(run_id)
        try:
            result = await self._client.chat_postMessage(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                text=text,
                blocks=_markdown_blocks(text),
                client_msg_id=_client_message_id(run_id, f"answer:{part_number}"),
            )
        except SlackApiError as exc:
            _raise_classified_delivery_error(exc)
        timestamp = str(result.get("ts", ""))
        if not timestamp:
            raise RuntimeError("Slack continuation response did not include a timestamp")
        await self._ledger.acknowledge_delivery_part(
            run_id,
            part_number=part_number,
            content_hash=content_hash,
            slack_message_ts=timestamp,
        )
        return timestamp

    async def _publish_terminal_notice(
        self,
        run_id: uuid.UUID,
        text: str,
        *,
        is_error: bool,
    ) -> None:
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.stream_state == SlackStreamState.OPENING:
            # Give the in-flight start owner time to persist Slack's returned timestamp. A
            # response-less start is ambiguous, so posting immediately could create duplicates.
            delivery = await self._wait_for_stream_open(run_id)
        stream_timestamp = delivery.stream_ts
        content_hash = _content_hash(text)
        metadata = _delivery_metadata(
            run_id,
            delivery_kind=f"terminal:{content_hash[:16]}",
            content_hash=content_hash,
        )
        terminal_chunks: list[dict[str, object]] = [
            _task_update(
                run_id,
                title="Work stopped" if is_error else "Work complete",
                status="error" if is_error else "complete",
                task_key="terminal",
                hide_title=True,
            ),
            {"type": "markdown_text", "text": text},
        ]
        if delivery.stream_state == SlackStreamState.OPEN:
            await self._ledger.transition_stream(
                run_id,
                expected_state=SlackStreamState.OPEN,
                target_state=SlackStreamState.STOPPING,
            )
            delivery = await self._ledger.get_delivery(run_id)
            if delivery.stream_ts is None:
                raise RunTransitionError(f"Stopping Slack stream has no timestamp: {run_id}")
            stream_timestamp = delivery.stream_ts

        delivered_timestamp: str | None = None
        if (
            delivery.stream_state
            in {
                SlackStreamState.STOPPING,
                SlackStreamState.UNCERTAIN,
                SlackStreamState.STOPPED,
            }
            and stream_timestamp is not None
        ):
            try:
                await self._client.chat_stopStream(
                    channel=delivery.channel_id,
                    ts=stream_timestamp,
                    chunks=terminal_chunks,
                    metadata=metadata,
                    session_status="active",
                )
                delivered_timestamp = stream_timestamp
            except SlackApiError as exc:
                if not _is_already_stopped_error(exc):
                    raise
                verification = await self._stream_contains_delivery(
                    run_id,
                    delivery,
                    timestamp=stream_timestamp,
                    metadata=metadata,
                )
                if verification is DeliveryVerification.UNKNOWN:
                    raise DeliveryReconciliationPendingError(
                        f"Slack terminal delivery could not be reconciled yet: {run_id}"
                    ) from exc
                if verification is DeliveryVerification.VERIFIED:
                    delivered_timestamp = stream_timestamp
            except Exception:
                with suppress(RunTransitionError):
                    await self._ledger.transition_stream(
                        run_id,
                        expected_state=delivery.stream_state,
                        target_state=SlackStreamState.UNCERTAIN,
                    )
                raise
            if delivery.stream_state in {
                SlackStreamState.STOPPING,
                SlackStreamState.UNCERTAIN,
            }:
                await self._ledger.transition_stream(
                    run_id,
                    expected_state=delivery.stream_state,
                    target_state=SlackStreamState.STOPPED,
                )
        elif delivery.stream_state != SlackStreamState.DEGRADED:
            await self._degrade_progress_surface(run_id)
            delivery = await self._ledger.get_delivery(run_id)

        if delivered_timestamp is None:
            if delivery.stream_state == SlackStreamState.NOT_STARTED:
                raise RunTransitionError(
                    f"Slack terminal delivery is deferred behind an earlier turn: {run_id}"
                )
            result = await self._client.chat_postMessage(
                channel=delivery.channel_id,
                thread_ts=delivery.thread_ts,
                text=text,
                blocks=_markdown_blocks(text),
                metadata=metadata,
                client_msg_id=_client_message_id(
                    run_id,
                    f"terminal:{content_hash[:16]}",
                ),
            )
            delivered_timestamp = str(result.get("ts", ""))
            if not delivered_timestamp:
                raise RuntimeError("Slack terminal response did not include a timestamp")
        delivery = await self._ledger.get_delivery(run_id)
        if delivery.response_ts is None:
            await self._ledger.set_response(run_id, delivered_timestamp)
        # A user's Stop click does not itself move the Agent Session away from processing.
        # Reasserting active is idempotent and covers both stream and immutable-post delivery.
        await self._set_session_status_strict(delivery, "active", run_id=run_id)

    async def _set_session_status_strict(
        self,
        delivery: DeliveryState,
        status: str,
        *,
        run_id: uuid.UUID,
    ) -> None:
        payload: dict[str, str] = {
            "channel_id": delivery.channel_id,
            "thread_ts": delivery.thread_ts,
            "status": status,
        }
        if status == "processing":
            payload["initiator_user_id"] = delivery.user_id
        try:
            await self._client.api_call("agents.sessions.setStatus", json=payload)
        except Exception as exc:
            logger.warning(
                "slack_agent_session_status_failed",
                agent_run_id=str(run_id),
                session_status=status,
                exception_class=type(exc).__name__,
            )
            raise

    async def _set_session_status_best_effort(
        self,
        delivery: DeliveryState,
        status: str,
        *,
        run_id: uuid.UUID,
    ) -> None:
        try:
            await self._set_session_status_strict(delivery, status, run_id=run_id)
        except Exception:
            return
