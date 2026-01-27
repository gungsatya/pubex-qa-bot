"""add slide image_path

Revision ID: 9b1f2d8c4a10
Revises: 8c2f0e8b9b17
Create Date: 2026-01-27 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "9b1f2d8c4a10"
down_revision: Union[str, None] = "8c2f0e8b9b17"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "slides",
        sa.Column("image_path", sa.String(), nullable=False, server_default=""),
    )
    op.drop_column("slides", "content_base_64")
    op.alter_column("slides", "image_path", server_default=None)


def downgrade() -> None:
    op.add_column(
        "slides",
        sa.Column("content_base_64", sa.Text(), nullable=True),
    )
    op.drop_column("slides", "image_path")
