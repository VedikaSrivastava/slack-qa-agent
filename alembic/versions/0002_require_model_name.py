"""Require every persisted run to record the code-defined model.

Revision ID: 0002
Revises: 0001
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        sa.text("UPDATE agent_runs SET model_name = 'gpt-4.1-mini' WHERE model_name IS NULL")
    )
    op.alter_column("agent_runs", "model_name", existing_type=sa.String(256), nullable=False)


def downgrade() -> None:
    op.alter_column("agent_runs", "model_name", existing_type=sa.String(256), nullable=True)
