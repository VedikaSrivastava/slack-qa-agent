"""Create application runtime ledger tables.

Revision ID: 0001
Revises: None
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
        sa.Column("status", sa.String(length=16), nullable=False),
        sa.Column("slack_team_id", sa.String(length=128), nullable=False),
        sa.Column("slack_channel_id", sa.String(length=128), nullable=False),
        sa.Column("slack_thread_ts", sa.String(length=64), nullable=False),
        sa.Column("slack_placeholder_ts", sa.String(length=64), nullable=True),
        sa.Column("slack_response_ts", sa.String(length=64), nullable=True),
        sa.Column("inngest_event_id", sa.String(length=512), nullable=True),
        sa.Column("prompt_version", sa.String(length=128), nullable=False),
        sa.Column("retrieval_version", sa.String(length=128), nullable=False),
        sa.Column("model_name", sa.String(length=256), nullable=True),
        sa.Column(
            "queued_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("queue_latency_ms", sa.Integer(), nullable=True),
        sa.Column("agent_latency_ms", sa.Integer(), nullable=True),
        sa.Column("total_latency_ms", sa.Integer(), nullable=True),
        sa.Column("tool_call_count", sa.Integer(), nullable=True),
        sa.Column("retrieval_round_count", sa.Integer(), nullable=True),
        sa.Column("input_tokens", sa.Integer(), nullable=True),
        sa.Column("output_tokens", sa.Integer(), nullable=True),
        sa.Column("estimated_cost_usd", sa.Numeric(12, 6), nullable=True),
        sa.Column("insufficient_evidence", sa.Boolean(), server_default=sa.false(), nullable=False),
        sa.Column("error_code", sa.String(length=128), nullable=True),
        sa.Column("sanitized_error_message", sa.Text(), nullable=True),
        sa.Column("result_json", sa.JSON(), nullable=True),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')",
            name="ck_agent_runs_status",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slack_event_id"),
    )
    op.create_index("ix_agent_runs_conversation_id", "agent_runs", ["conversation_id"])
    op.create_index("ix_agent_runs_status_queued_at", "agent_runs", ["status", "queued_at"])

    op.create_table(
        "run_sources",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("artifact_id", sa.String(length=512), nullable=False),
        sa.Column("artifact_title", sa.Text(), nullable=False),
        sa.Column("retrieval_rank", sa.Integer(), nullable=False),
        sa.Column("retrieval_score", sa.Numeric(18, 8), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("agent_run_id", "artifact_id", name="uq_run_sources_run_artifact"),
    )
    op.create_index("ix_run_sources_agent_run_id", "run_sources", ["agent_run_id"])

    op.create_table(
        "feedback",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("agent_run_id", sa.Uuid(), nullable=False),
        sa.Column("rating", sa.Integer(), nullable=False),
        sa.Column("comment", sa.Text(), nullable=True),
        sa.Column(
            "created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False
        ),
        sa.CheckConstraint("rating IN (-1, 1)", name="ck_feedback_rating"),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_feedback_agent_run_id", "feedback", ["agent_run_id"])


def downgrade() -> None:
    op.drop_index("ix_feedback_agent_run_id", table_name="feedback")
    op.drop_table("feedback")
    op.drop_index("ix_run_sources_agent_run_id", table_name="run_sources")
    op.drop_table("run_sources")
    op.drop_index("ix_agent_runs_status_queued_at", table_name="agent_runs")
    op.drop_index("ix_agent_runs_conversation_id", table_name="agent_runs")
    op.drop_table("agent_runs")
