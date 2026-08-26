"""SQLAlchemy models for the application-owned runtime ledger."""

from __future__ import annotations

import uuid
from datetime import datetime
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


class AgentRun(Base):
    __tablename__ = "agent_runs"
    __table_args__ = (
        CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_agent_runs_status",
        ),
        Index("ix_agent_runs_conversation_id", "conversation_id"),
        Index("ix_agent_runs_status_queued_at", "status", "queued_at"),
    )

    id: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True, default=uuid.uuid4)
    slack_event_id: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    conversation_id: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default=RunStatus.QUEUED.value)
    slack_team_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_channel_id: Mapped[str] = mapped_column(String(128), nullable=False)
    slack_thread_ts: Mapped[str] = mapped_column(String(64), nullable=False)
    slack_placeholder_ts: Mapped[str | None] = mapped_column(String(64))
    slack_response_ts: Mapped[str | None] = mapped_column(String(64))
    inngest_event_id: Mapped[str | None] = mapped_column(String(512))
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


class RunSource(Base):
    __tablename__ = "run_sources"
    __table_args__ = (
        UniqueConstraint("agent_run_id", "artifact_id", name="uq_run_sources_run_artifact"),
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
