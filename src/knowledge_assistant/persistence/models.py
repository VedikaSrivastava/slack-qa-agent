"""SQLAlchemy models for the application-owned runtime ledger."""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    text,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SlackStreamState(StrEnum):
    NOT_STARTED = "not_started"
    OPENING = "opening"
    OPEN = "open"
    STOPPING = "stopping"
    STOPPED = "stopped"
    UNCERTAIN = "uncertain"
    DEGRADED = "degraded"


class SlackStreamMode(StrEnum):
    CHUNKS = "chunks"


class DeliveryStatus(StrEnum):
    PENDING = "pending"
    DELIVERING = "delivering"
    DELIVERED = "delivered"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SlackTurnKind(StrEnum):
    EXPLICIT_MENTION = "explicit_mention"
    FOLLOW_UP = "follow_up"


class SlackTurnStatus(StrEnum):
    PENDING = "pending"
    PROCESSING = "processing"
    ROUTED = "routed"
    SUPPRESSED = "suppressed"
    FAILED = "failed"


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_agent_runs_status",
        ),
        CheckConstraint(
            "slack_stream_state IN "
            "('not_started', 'opening', 'open', 'stopping', 'stopped', "
            "'uncertain', 'degraded')",
            name="ck_agent_runs_slack_stream_state",
        ),
        CheckConstraint(
            "slack_stream_mode IS NULL OR slack_stream_mode = 'chunks'",
            name="ck_agent_runs_slack_stream_mode",
        ),
        CheckConstraint(
            "delivery_status IN ('pending', 'delivering', 'delivered', 'failed', 'cancelled')",
            name="ck_agent_runs_delivery_status",
        ),
        CheckConstraint(
            "last_progress_sequence >= 0",
            name="ck_agent_runs_last_progress_sequence",
        ),
        CheckConstraint(
            "delivery_manifest_version IS NULL OR delivery_manifest_version > 0",
            name="ck_agent_runs_delivery_manifest_version",
        ),
        CheckConstraint(
            "(delivery_manifest_version IS NULL AND delivery_manifest_hash IS NULL) OR "
            "(delivery_manifest_version IS NOT NULL "
            "AND delivery_manifest_hash ~ '^[0-9a-f]{64}$')",
            name="ck_agent_runs_delivery_manifest_pair",
        ),
        CheckConstraint(
            "queue_latency_ms IS NULL OR queue_latency_ms >= 0",
            name="ck_agent_runs_queue_latency",
        ),
        CheckConstraint(
            "agent_latency_ms IS NULL OR agent_latency_ms >= 0",
            name="ck_agent_runs_agent_latency",
        ),
        CheckConstraint(
            "total_latency_ms IS NULL OR total_latency_ms >= 0",
            name="ck_agent_runs_total_latency",
        ),
        CheckConstraint(
            "tool_call_count IS NULL OR tool_call_count >= 0",
            name="ck_agent_runs_tool_calls",
        ),
        CheckConstraint(
            "model_call_count IS NULL OR model_call_count >= 0",
            name="ck_agent_runs_model_calls",
        ),
        CheckConstraint(
            "retrieval_round_count IS NULL OR retrieval_round_count BETWEEN 0 AND 2",
            name="ck_agent_runs_retrieval_rounds",
        ),
        CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_agent_runs_input_tokens",
        ),
        CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_agent_runs_output_tokens",
        ),
        CheckConstraint(
            "(status = 'queued' AND started_at IS NULL AND completed_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL AND completed_at IS NULL) OR "
            "(status IN ('succeeded', 'failed', 'cancelled') AND completed_at IS NOT NULL)",
            name="ck_agent_runs_lifecycle_timestamps",
        ),
        CheckConstraint(
            "status <> 'succeeded' OR "
            "(started_at IS NOT NULL AND result_json IS NOT NULL "
            "AND delivery_status = 'delivered' AND cancellation_requested = false)",
            name="ck_agent_runs_succeeded_state",
        ),
        CheckConstraint(
            "status <> 'cancelled' OR "
            "(cancellation_requested = true AND delivery_status = 'cancelled')",
            name="ck_agent_runs_cancelled_state",
        ),
        CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL AND sanitized_error_message IS NULL)",
            name="ck_agent_runs_error_state",
        ),
        CheckConstraint(
            "slack_stream_state <> 'not_started' OR "
            "(slack_stream_mode IS NULL AND slack_stream_ts IS NULL)",
            name="ck_agent_runs_not_started_stream",
        ),
        CheckConstraint(
            "slack_stream_state <> 'opening' OR "
            "(slack_stream_mode IS NOT NULL AND slack_stream_ts IS NULL)",
            name="ck_agent_runs_opening_stream",
        ),
        CheckConstraint(
            "slack_stream_state NOT IN ('open', 'stopping', 'stopped') OR "
            "(slack_stream_mode IS NOT NULL AND slack_stream_ts IS NOT NULL)",
            name="ck_agent_runs_identified_stream",
        ),
        CheckConstraint(
            "delivery_status NOT IN ('delivering', 'delivered') OR "
            "delivery_manifest_version IS NOT NULL",
            name="ck_agent_runs_delivery_has_manifest",
        ),
        CheckConstraint(
            "delivery_status <> 'cancelled' OR cancellation_requested = true",
            name="ck_agent_runs_cancelled_delivery",
        ),
        UniqueConstraint("slack_event_id", name="uq_agent_runs_slack_event_id"),
        Index("ix_agent_runs_conversation_id", "conversation_id"),
        Index(
            "uq_agent_runs_active_progress_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text(
                "status IN ('queued', 'running') AND slack_stream_state <> 'not_started'"
            ),
        ),
        Index(
            "ix_agent_runs_slack_thread",
            "slack_team_id",
            "slack_channel_id",
            "slack_thread_ts",
        ),
        Index("ix_agent_runs_status_queued_at", "status", "queued_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slack_event_id: Mapped[str] = mapped_column(
        String(512),
        nullable=False,
    )
    conversation_id: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=RunStatus.QUEUED.value,
        server_default=RunStatus.QUEUED.value,
    )
    slack_team_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_message_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_thread_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_response_ts: Mapped[str | None] = mapped_column(String(64))
    slack_stream_state: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=SlackStreamState.NOT_STARTED.value,
        server_default=SlackStreamState.NOT_STARTED.value,
    )
    slack_stream_mode: Mapped[str | None] = mapped_column(String(16))
    slack_stream_ts: Mapped[str | None] = mapped_column(String(64))
    last_progress_sequence: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=0,
        server_default="0",
    )
    delivery_status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=DeliveryStatus.PENDING.value,
        server_default=DeliveryStatus.PENDING.value,
    )
    delivery_manifest_version: Mapped[int | None] = mapped_column(Integer)
    delivery_manifest_hash: Mapped[str | None] = mapped_column(String(64))
    cancellation_requested: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    prompt_version: Mapped[str] = mapped_column(String(128), nullable=False)
    retrieval_version: Mapped[str] = mapped_column(String(128), nullable=False)
    model_name: Mapped[str] = mapped_column(String(256), nullable=False)
    queued_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    queue_latency_ms: Mapped[int | None] = mapped_column(Integer)
    agent_latency_ms: Mapped[int | None] = mapped_column(Integer)
    total_latency_ms: Mapped[int | None] = mapped_column(Integer)
    tool_call_count: Mapped[int | None] = mapped_column(Integer)
    model_call_count: Mapped[int | None] = mapped_column(Integer)
    retrieval_round_count: Mapped[int | None] = mapped_column(Integer)
    input_tokens: Mapped[int | None] = mapped_column(Integer)
    output_tokens: Mapped[int | None] = mapped_column(Integer)
    insufficient_evidence: Mapped[bool] = mapped_column(
        Boolean, nullable=False, default=False, server_default="false"
    )
    error_code: Mapped[str | None] = mapped_column(String(128))
    sanitized_error_message: Mapped[str | None] = mapped_column(Text)
    result_json: Mapped[dict[str, Any] | None] = mapped_column(JSON)

    sources: Mapped[list[RunSource]] = relationship(
        back_populates="agent_run", cascade="all, delete-orphan"
    )
    delivery_parts: Mapped[list[RunDeliveryPart]] = relationship(
        back_populates="agent_run",
        cascade="all, delete-orphan",
        order_by="RunDeliveryPart.part_number",
    )


class SlackTurn(Base):
    """Durable causal queue entry for one immutable Slack message event."""

    __tablename__ = "slack_turns"
    __table_args__ = (
        CheckConstraint(
            "kind IN ('explicit_mention', 'follow_up')",
            name="ck_slack_turns_kind",
        ),
        CheckConstraint(
            "status IN ('pending', 'processing', 'routed', 'suppressed', 'failed')",
            name="ck_slack_turns_status",
        ),
        CheckConstraint(
            "message_ts_value >= 0",
            name="ck_slack_turns_message_ts_value",
        ),
        CheckConstraint(
            "CAST(slack_message_ts AS NUMERIC) = message_ts_value",
            name="ck_slack_turns_message_ts_matches_value",
        ),
        CheckConstraint(
            "conversation_id = slack_team_id || ':' || slack_channel_id || ':' || slack_thread_ts",
            name="ck_slack_turns_conversation_identity",
        ),
        CheckConstraint(
            "(status = 'pending' AND claimed_at IS NULL AND completed_at IS NULL "
            "AND agent_run_id IS NULL) OR "
            "(status = 'processing' AND claimed_at IS NOT NULL AND completed_at IS NULL) OR "
            "(status = 'routed' AND claimed_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND agent_run_id IS NOT NULL) OR "
            "(status IN ('suppressed', 'failed') AND claimed_at IS NOT NULL "
            "AND completed_at IS NOT NULL AND agent_run_id IS NULL)",
            name="ck_slack_turns_lifecycle",
        ),
        CheckConstraint(
            "claimed_at IS NULL OR claimed_at >= created_at",
            name="ck_slack_turns_claimed_after_created",
        ),
        CheckConstraint(
            "completed_at IS NULL OR completed_at >= claimed_at",
            name="ck_slack_turns_completed_after_claimed",
        ),
        UniqueConstraint("agent_run_id", name="uq_slack_turns_agent_run_id"),
        Index(
            "ix_slack_turns_causal_head",
            "conversation_id",
            "message_ts_value",
            "created_at",
            "event_id",
            postgresql_where=text("status IN ('pending', 'processing')"),
        ),
        Index(
            "uq_slack_turns_processing_conversation",
            "conversation_id",
            unique=True,
            postgresql_where=text("status = 'processing'"),
        ),
    )

    event_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    slack_team_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_message_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    message_ts_value: Mapped[Decimal] = mapped_column(Numeric(30, 6), nullable=False)
    slack_thread_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    conversation_id: Mapped[str] = mapped_column(String(512), nullable=False)
    kind: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(
        String(16),
        nullable=False,
        default=SlackTurnStatus.PENDING.value,
        server_default=SlackTurnStatus.PENDING.value,
    )
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("agent_runs.id", ondelete="RESTRICT"),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    claimed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class SlackStopEvent(Base):
    """Idempotency record binding one Slack Stop event to its original outcome."""

    __tablename__ = "slack_stop_events"
    __table_args__ = (
        CheckConstraint(
            "agent_run_id IS NOT NULL OR accepted = false",
            name="ck_slack_stop_events_accepted_run",
        ),
    )

    event_id: Mapped[str] = mapped_column(String(512), primary_key=True)
    slack_team_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_user_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_thread_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_event_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    agent_run_id: Mapped[uuid.UUID | None] = mapped_column(
        Uuid,
        ForeignKey("agent_runs.id", ondelete="RESTRICT"),
    )
    accepted: Mapped[bool] = mapped_column(
        Boolean,
        nullable=False,
        default=False,
        server_default="false",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    stopped_streams: Mapped[list[SlackStoppedStream]] = relationship(
        back_populates="stop_event",
        cascade="all, delete-orphan",
        order_by="SlackStoppedStream.stream_order",
    )


class SlackStoppedStream(Base):
    """One atomic streaming-message identity reported by one Slack Stop event."""

    __tablename__ = "slack_stopped_streams"
    __table_args__ = (
        CheckConstraint(
            "stream_order > 0",
            name="ck_slack_stopped_streams_stream_order",
        ),
        UniqueConstraint(
            "event_id",
            "slack_message_ts",
            name="uq_slack_stopped_streams_event_timestamp",
        ),
    )

    event_id: Mapped[str] = mapped_column(
        String(512),
        ForeignKey("slack_stop_events.event_id", ondelete="CASCADE"),
        primary_key=True,
    )
    stream_order: Mapped[int] = mapped_column(Integer, primary_key=True)
    slack_message_ts: Mapped[str] = mapped_column(String(64), nullable=False)

    stop_event: Mapped[SlackStopEvent] = relationship(back_populates="stopped_streams")


class RunSource(Base):
    __tablename__ = "run_sources"
    __table_args__ = (
        UniqueConstraint("agent_run_id", "artifact_id", name="uq_run_sources_run_artifact"),
        UniqueConstraint("agent_run_id", "retrieval_rank", name="uq_run_sources_run_rank"),
        CheckConstraint("retrieval_rank > 0", name="ck_run_sources_retrieval_rank"),
        Index("ix_run_sources_agent_run_id", "agent_run_id"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid, ForeignKey("agent_runs.id", ondelete="CASCADE"), nullable=False
    )
    artifact_id: Mapped[str] = mapped_column(String(512), nullable=False)
    artifact_title: Mapped[str] = mapped_column(Text, nullable=False)
    retrieval_rank: Mapped[int] = mapped_column(Integer, nullable=False)
    retrieval_score: Mapped[float | None] = mapped_column(Numeric(18, 8))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    agent_run: Mapped[AgentRun] = relationship(back_populates="sources")


class RunDeliveryPart(Base):
    __tablename__ = "run_delivery_parts"
    __table_args__ = (
        CheckConstraint("part_number > 0", name="ck_run_delivery_parts_part_number"),
        CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_run_delivery_parts_content_hash",
        ),
        CheckConstraint(
            "(slack_message_ts IS NULL AND acknowledged_at IS NULL) OR "
            "(slack_message_ts IS NOT NULL AND acknowledged_at IS NOT NULL)",
            name="ck_run_delivery_parts_ack_pair",
        ),
    )

    agent_run_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("agent_runs.id", ondelete="CASCADE"),
        primary_key=True,
    )
    part_number: Mapped[int] = mapped_column(Integer, primary_key=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_message_ts: Mapped[str | None] = mapped_column(String(64))
    acknowledged_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    agent_run: Mapped[AgentRun] = relationship(back_populates="delivery_parts")
