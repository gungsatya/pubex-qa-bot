"""seed document_status lifecycle

Revision ID: 53752c53f3ee
Revises: 623e1df0c15e
Create Date: 2026-01-22 16:55:33.618467

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = '53752c53f3ee'
down_revision: Union[str, None] = '623e1df0c15e'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.execute("""
        INSERT INTO document_status (id, name) VALUES
            (1, 'downloaded'),
            (2, 'failed_parsed'),
            (3, 'parsed'),
            (4, 'failed_embedded'),
            (5, 'embedded'),
            (6, 'ready')
        ON CONFLICT DO NOTHING;
    """)


def downgrade():
    op.execute("""
        DELETE FROM document_status
        WHERE id IN (1,2,3,4,5,6);
    """)
