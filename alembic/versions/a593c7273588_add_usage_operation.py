"""add usage operation

Revision ID: a593c7273588
Revises: 086da34b695d
Create Date: <KEEP_GENERATED_DATE>
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "a593c7273588"
down_revision: Union[str, Sequence[str], None] = "086da34b695d"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add the new operation column temporarily as nullable so
    # existing usage_records can be backfilled safely.
    op.add_column(
        "usage_records",
        sa.Column(
            "operation",
            sa.String(length=20),
            nullable=True,
        ),
    )

    # Existing usage records were created by /chat, so they represent
    # generation operations.
    op.execute(
        "UPDATE usage_records SET operation = 'generate'"
    )

    # Enforce the new invariant for all future rows.
    op.alter_column(
        "usage_records",
        "operation",
        nullable=False,
    )

    # request_id used to be globally unique.
    op.drop_index(
        "ix_usage_records_request_id",
        table_name="usage_records",
    )

    # A request may now contain one embed operation and one generate
    # operation, while each operation remains unique within the request.
    op.create_unique_constraint(
        "uq_usage_records_request_operation",
        "usage_records",
        ["request_id", "operation"],
    )


def downgrade() -> None:
    # WARNING:
    # Downgrade is only safe if no request_id has multiple operations.
    op.drop_constraint(
        "uq_usage_records_request_operation",
        "usage_records",
        type_="unique",
    )

    op.create_index(
        "ix_usage_records_request_id",
        "usage_records",
        ["request_id"],
        unique=True,
    )

    op.drop_column(
        "usage_records",
        "operation",
    )