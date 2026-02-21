"""add documents file_md_path

Revision ID: b41e8c9d2f33
Revises: 9b1f2d8c4a10
Create Date: 2026-02-20 00:00:00.000000

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "b41e8c9d2f33"
down_revision: Union[str, None] = "9b1f2d8c4a10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "documents",
        sa.Column("file_md_path", sa.String(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("documents", "file_md_path")
