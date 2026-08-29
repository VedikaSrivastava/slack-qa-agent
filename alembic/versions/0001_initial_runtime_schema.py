"""Create the complete application-owned runtime ledger.

Revision ID: 0001
Revises: None

This is the baseline for an unreleased application. It intentionally contains
the complete current schema instead of preserving pre-release migration history.
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("slack_event_id", sa.String(length=512), nullable=False),
        sa.Column("conversation_id", sa.String(length=512), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="queued",
            nullable=False,
        ),
        sa.Column("slack_team_id", sa.String(length=128), nullable=False),
        sa.Column("slack_channel_id", sa.String(length=128), nullable=False),
        sa.Column("slack_user_id", sa.String(length=128), nullable=False),
        sa.Column("slack_message_ts", sa.String(length=64), nullable=False),
        sa.Column("slack_thread_ts", sa.String(length=64), nullable=False),
        sa.Column("slack_response_ts", sa.String(length=64), nullable=True),
        sa.Column(
            "slack_stream_state",
            sa.String(length=16),
            server_default="not_started",
            nullable=False,
        ),
        sa.Column("slack_stream_mode", sa.String(length=16), nullable=True),
        sa.Column("slack_stream_ts", sa.String(length=64), nullable=True),
        sa.Column(
            "last_progress_sequence",
            sa.Integer(),
            server_default="0",
            nullable=False,
        ),
        sa.Column(
            "delivery_status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("delivery_manifest_version", sa.Integer(), nullable=True),
        sa.Column("delivery_manifest_hash", sa.String(length=64), nullable=True),
        sa.Column(
            "cancellation_requested",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("prompt_version", sa.String(length=128), nullable=False),
        sa.Column("retrieval_version", sa.String(length=128), nullable=False),
        sa.Column("model_name", sa.String(length=256), nullable=False),
        sa.Column(
            "queued_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queue_latency_ms", sa.Integer(), nullable=True),
        sa.Column("agent_latency_ms", sa.Integer(), nullable=True),
        sa.Column("total_latency_ms", sa.Integer(), nullable=True),
        sa.Column("tool_call_count", sa.Integer(), nullable=True),
        sa.Column("model_call_count", sa.Integer(), nullable=True),
        sa.Column("retrieval_round_count", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column(
            "insufficient_evidence",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("sanitized_error_message", sa.Text(), nullable=True),
        # The immutable typed response snapshot makes a lost Inngest acknowledgement
        # replayable without calling the model again. Queryable sources remain normalized.
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_agent_runs_status",
        ),
        sa.CheckConstraint(
            "slack_stream_state IN "
            "('not_started', 'opening', 'open', 'stopping', 'stopped', "
            "'uncertain', 'degraded')",
            name="ck_agent_runs_slack_stream_state",
        ),
        sa.CheckConstraint(
            "slack_stream_mode IS NULL OR slack_stream_mode = 'chunks'",
            name="ck_agent_runs_slack_stream_mode",
        ),
        sa.CheckConstraint(
            "delivery_status IN ('pending', 'delivering', 'delivered', 'failed', 'cancelled')",
            name="ck_agent_runs_delivery_status",
        ),
        sa.CheckConstraint(
            "last_progress_sequence >= 0",
            name="ck_agent_runs_last_progress_sequence",
        ),
        sa.CheckConstraint(
            "delivery_manifest_version IS NULL OR delivery_manifest_version > 0",
            name="ck_agent_runs_delivery_manifest_version",
        ),
        sa.CheckConstraint(
            "(delivery_manifest_version IS NULL AND delivery_manifest_hash IS NULL) OR "
            "(delivery_manifest_version IS NOT NULL "
            "AND delivery_manifest_hash ~ '^[0-9a-f]{64}$')",
            name="ck_agent_runs_delivery_manifest_pair",
        ),
        sa.CheckConstraint(
            "queue_latency_ms IS NULL OR queue_latency_ms >= 0",
            name="ck_agent_runs_queue_latency",
        ),
        sa.CheckConstraint(
            "agent_latency_ms IS NULL OR agent_latency_ms >= 0",
            name="ck_agent_runs_agent_latency",
        ),
        sa.CheckConstraint(
            "total_latency_ms IS NULL OR total_latency_ms >= 0",
            name="ck_agent_runs_total_latency",
        ),
        sa.CheckConstraint(
            "tool_call_count IS NULL OR tool_call_count >= 0",
            name="ck_agent_runs_tool_calls",
        ),
        sa.CheckConstraint(
            "model_call_count IS NULL OR model_call_count >= 0",
            name="ck_agent_runs_model_calls",
        ),
        sa.CheckConstraint(
            "retrieval_round_count IS NULL OR retrieval_round_count BETWEEN 0 AND 2",
            name="ck_agent_runs_retrieval_rounds",
        ),
        sa.CheckConstraint(
            "input_tokens IS NULL OR input_tokens >= 0",
            name="ck_agent_runs_input_tokens",
        ),
        sa.CheckConstraint(
            "output_tokens IS NULL OR output_tokens >= 0",
            name="ck_agent_runs_output_tokens",
        ),
        sa.CheckConstraint(
            "(status = 'queued' AND started_at IS NULL AND completed_at IS NULL) OR "
            "(status = 'running' AND started_at IS NOT NULL AND completed_at IS NULL) OR "
            "(status IN ('succeeded', 'failed', 'cancelled') AND completed_at IS NOT NULL)",
            name="ck_agent_runs_lifecycle_timestamps",
        ),
        sa.CheckConstraint(
            "status <> 'succeeded' OR "
            "(started_at IS NOT NULL AND result_json IS NOT NULL "
            "AND delivery_status = 'delivered' AND cancellation_requested = false)",
            name="ck_agent_runs_succeeded_state",
        ),
        sa.CheckConstraint(
            "status <> 'cancelled' OR "
            "(cancellation_requested = true AND delivery_status = 'cancelled')",
            name="ck_agent_runs_cancelled_state",
        ),
        sa.CheckConstraint(
            "(status = 'failed' AND error_code IS NOT NULL) OR "
            "(status <> 'failed' AND error_code IS NULL "
            "AND sanitized_error_message IS NULL)",
            name="ck_agent_runs_error_state",
        ),
        sa.CheckConstraint(
            "slack_stream_state <> 'not_started' OR "
            "(slack_stream_mode IS NULL AND slack_stream_ts IS NULL)",
            name="ck_agent_runs_not_started_stream",
        ),
        sa.CheckConstraint(
            "slack_stream_state <> 'opening' OR "
            "(slack_stream_mode IS NOT NULL AND slack_stream_ts IS NULL)",
            name="ck_agent_runs_opening_stream",
        ),
        sa.CheckConstraint(
            "slack_stream_state NOT IN ('open', 'stopping', 'stopped') OR "
            "(slack_stream_mode IS NOT NULL AND slack_stream_ts IS NOT NULL)",
            name="ck_agent_runs_identified_stream",
        ),
        sa.CheckConstraint(
            "delivery_status NOT IN ('delivering', 'delivered') OR "
            "delivery_manifest_version IS NOT NULL",
            name="ck_agent_runs_delivery_has_manifest",
        ),
        sa.CheckConstraint(
            "delivery_status <> 'cancelled' OR cancellation_requested = true",
            name="ck_agent_runs_cancelled_delivery",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slack_event_id", name="uq_agent_runs_slack_event_id"),
    )
    op.create_index(
        "ix_agent_runs_conversation_id",
        "agent_runs",
        ["conversation_id"],
    )
    op.create_index(
        "ix_agent_runs_slack_thread",
        "agent_runs",
        ["slack_team_id", "slack_channel_id", "slack_thread_ts"],
    )
    op.create_index(
        "ix_agent_runs_status_queued_at",
        "agent_runs",
        ["status", "queued_at"],
    )
    op.create_index(
        "uq_agent_runs_active_progress_conversation",
        "agent_runs",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text(
            "status IN ('queued', 'running') AND slack_stream_state <> 'not_started'"
        ),
    )

    op.create_table(
        "slack_turns",
        sa.Column("event_id", sa.String(length=512), nullable=False),
        sa.Column("slack_team_id", sa.String(length=128), nullable=False),
        sa.Column("slack_channel_id", sa.String(length=128), nullable=False),
        sa.Column("slack_user_id", sa.String(length=128), nullable=False),
        sa.Column("slack_message_ts", sa.String(length=64), nullable=False),
        sa.Column("message_ts_value", sa.Numeric(30, 6), nullable=False),
        sa.Column("slack_thread_ts", sa.String(length=64), nullable=False),
        sa.Column("conversation_id", sa.String(length=512), nullable=False),
        sa.Column("kind", sa.String(length=32), nullable=False),
        sa.Column(
            "status",
            sa.String(length=16),
            server_default="pending",
            nullable=False,
        ),
        sa.Column("agent_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.Column("claimed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.CheckConstraint(
            "kind IN ('explicit_mention', 'follow_up')",
            name="ck_slack_turns_kind",
        ),
        sa.CheckConstraint(
            "status IN ('pending', 'processing', 'routed', 'suppressed', 'failed')",
            name="ck_slack_turns_status",
        ),
        sa.CheckConstraint(
            "message_ts_value >= 0",
            name="ck_slack_turns_message_ts_value",
        ),
        sa.CheckConstraint(
            "CAST(slack_message_ts AS NUMERIC) = message_ts_value",
            name="ck_slack_turns_message_ts_matches_value",
        ),
        sa.CheckConstraint(
            "conversation_id = slack_team_id || ':' || slack_channel_id || ':' || slack_thread_ts",
            name="ck_slack_turns_conversation_identity",
        ),
        sa.CheckConstraint(
            "(status = 'pending' AND claimed_at IS NULL AND completed_at IS NULL "
            "AND agent_run_id IS NULL) OR "
            "(status = 'processing' AND claimed_at IS NOT NULL AND completed_at IS NULL) OR "
            "(status = 'routed' AND claimed_at IS NOT NULL AND completed_at IS NOT NULL "
            "AND agent_run_id IS NOT NULL) OR "
            "(status IN ('suppressed', 'failed') AND claimed_at IS NOT NULL "
            "AND completed_at IS NOT NULL AND agent_run_id IS NULL)",
            name="ck_slack_turns_lifecycle",
        ),
        sa.CheckConstraint(
            "claimed_at IS NULL OR claimed_at >= created_at",
            name="ck_slack_turns_claimed_after_created",
        ),
        sa.CheckConstraint(
            "completed_at IS NULL OR completed_at >= claimed_at",
            name="ck_slack_turns_completed_after_claimed",
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("event_id"),
        sa.UniqueConstraint("agent_run_id", name="uq_slack_turns_agent_run_id"),
    )
    op.create_index(
        "ix_slack_turns_causal_head",
        "slack_turns",
        ["conversation_id", "message_ts_value", "created_at", "event_id"],
        postgresql_where=sa.text("status IN ('pending', 'processing')"),
    )
    op.create_index(
        "uq_slack_turns_processing_conversation",
        "slack_turns",
        ["conversation_id"],
        unique=True,
        postgresql_where=sa.text("status = 'processing'"),
    )

    op.create_table(
        "run_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.String(length=512), nullable=False),
        sa.Column("artifact_title", sa.Text(), nullable=False),
        sa.Column("retrieval_rank", sa.Integer(), nullable=False),
        sa.Column("retrieval_score", sa.Numeric(18, 8), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "retrieval_rank > 0",
            name="ck_run_sources_retrieval_rank",
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "agent_run_id",
            "artifact_id",
            name="uq_run_sources_run_artifact",
        ),
        sa.UniqueConstraint(
            "agent_run_id",
            "retrieval_rank",
            name="uq_run_sources_run_rank",
        ),
    )
    op.create_index(
        "ix_run_sources_agent_run_id",
        "run_sources",
        ["agent_run_id"],
    )

    op.create_table(
        "slack_stop_events",
        sa.Column("event_id", sa.String(length=512), nullable=False),
        sa.Column("slack_team_id", sa.String(length=128), nullable=False),
        sa.Column("slack_channel_id", sa.String(length=128), nullable=False),
        sa.Column("slack_user_id", sa.String(length=128), nullable=False),
        sa.Column("slack_thread_ts", sa.String(length=64), nullable=False),
        sa.Column("slack_event_ts", sa.String(length=64), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=True),
        sa.Column(
            "accepted",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "agent_run_id IS NOT NULL OR accepted = false",
            name="ck_slack_stop_events_accepted_run",
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="RESTRICT",
        ),
        sa.PrimaryKeyConstraint("event_id"),
    )

    op.create_table(
        "slack_stopped_streams",
        sa.Column("event_id", sa.String(length=512), nullable=False),
        sa.Column("stream_order", sa.Integer(), nullable=False),
        sa.Column("slack_message_ts", sa.String(length=64), nullable=False),
        sa.CheckConstraint(
            "stream_order > 0",
            name="ck_slack_stopped_streams_stream_order",
        ),
        sa.ForeignKeyConstraint(
            ["event_id"],
            ["slack_stop_events.event_id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("event_id", "stream_order"),
        sa.UniqueConstraint(
            "event_id",
            "slack_message_ts",
            name="uq_slack_stopped_streams_event_timestamp",
        ),
    )

    op.create_table(
        "run_delivery_parts",
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("part_number", sa.Integer(), nullable=False),
        sa.Column("content_hash", sa.String(length=64), nullable=False),
        sa.Column("slack_message_ts", sa.String(length=64), nullable=True),
        sa.Column("acknowledged_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.CheckConstraint(
            "part_number > 0",
            name="ck_run_delivery_parts_part_number",
        ),
        sa.CheckConstraint(
            "content_hash ~ '^[0-9a-f]{64}$'",
            name="ck_run_delivery_parts_content_hash",
        ),
        sa.CheckConstraint(
            "(slack_message_ts IS NULL AND acknowledged_at IS NULL) OR "
            "(slack_message_ts IS NOT NULL AND acknowledged_at IS NOT NULL)",
            name="ck_run_delivery_parts_ack_pair",
        ),
        sa.ForeignKeyConstraint(
            ["agent_run_id"],
            ["agent_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("agent_run_id", "part_number"),
    )


def downgrade() -> None:
    op.drop_table("run_delivery_parts")
    op.drop_table("slack_stopped_streams")
    op.drop_table("slack_stop_events")
    op.drop_index("ix_run_sources_agent_run_id", table_name="run_sources")
    op.drop_table("run_sources")
    op.drop_index(
        "uq_slack_turns_processing_conversation",
        table_name="slack_turns",
    )
    op.drop_index("ix_slack_turns_causal_head", table_name="slack_turns")
    op.drop_table("slack_turns")
    op.drop_index(
        "uq_agent_runs_active_progress_conversation",
        table_name="agent_runs",
    )
    op.drop_index("ix_agent_runs_status_queued_at", table_name="agent_runs")
    op.drop_index("ix_agent_runs_slack_thread", table_name="agent_runs")
    op.drop_index("ix_agent_runs_conversation_id", table_name="agent_runs")
    op.drop_table("agent_runs")
