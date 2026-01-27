"""add slide ingestion timestamps

Revision ID: 8c2f0e8b9b17
Revises: 53752c53f3ee
Create Date: 2026-01-26 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8c2f0e8b9b17"
down_revision: Union[str, None] = "53752c53f3ee"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "slides",
        sa.Column("ingestion_start_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )
    op.add_column(
        "slides",
        sa.Column("ingestion_end_at", sa.TIMESTAMP(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("slides", "ingestion_end_at")
    op.drop_column("slides", "ingestion_start_at")
