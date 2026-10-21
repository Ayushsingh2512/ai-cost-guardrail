"""remove orphaned budget status column

Revision ID: 3f4fef3dda06
Revises: f9540b35ac35
Create Date: 2026-10-05 23:35:16.098481

"""

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "3f4fef3dda06"
down_revision: str | Sequence[str] | None = "f9540b35ac35"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("usage_records")}

    if "budget_status" in columns:
        op.drop_column("usage_records", "budget_status")


def downgrade() -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    columns = {col["name"] for col in inspector.get_columns("usage_records")}

    if "budget_status" not in columns:
        op.add_column(
            "usage_records",
            sa.Column(
                "budget_status",
                sa.String(length=30),
                nullable=False,
            ),
        )
