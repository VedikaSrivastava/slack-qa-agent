"""Persist bounded Slack turn text for explicit follow-up context."""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "slack_turns",
        sa.Column("message_text", sa.Text(), server_default="", nullable=False),
    )
    op.alter_column("slack_turns", "message_text", server_default=None)


def downgrade() -> None:
    op.drop_column("slack_turns", "message_text")
