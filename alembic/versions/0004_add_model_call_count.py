"""Persist model-call accounting for completed agent runs.

Revision ID: 0004
Revises: 0003
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: str | None = "0003"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("model_call_count", sa.Integer(), nullable=True))


def downgrade() -> None:
    op.drop_column("agent_runs", "model_call_count")
