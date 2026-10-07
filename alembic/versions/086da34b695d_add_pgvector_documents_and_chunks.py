"""add pgvector documents and chunks

Revision ID: 086da34b695d
Revises: 3f4fef3dda06
Create Date: 2026-10-07 09:41:29.774224

"""

from typing import Sequence, Union

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = "086da34b695d"
down_revision: Union[str, Sequence[str], None] = "3f4fef3dda06"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "documents",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            primary_key=True,
            nullable=False,
        ),
        sa.Column(
            "tenant_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "source",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id"],
            ["tenants.id"],
            ondelete="CASCADE",
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_documents_tenant_id_id",
        ),
    )


    op.create_table(
        "document_chunks",
        sa.Column(
            "id",
            sa.Integer(),
            primary_key=True,
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "tenant_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "document_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "chunk_index",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "page",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "text",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "metadata",
            postgresql.JSONB(),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "embedding",
            pgvector.sqlalchemy.Vector(768),
            nullable=False,
        ),
        sa.Column(
            "fingerprint",
            sa.Text(),
            nullable=False,
        ),
        sa.UniqueConstraint(
            "tenant_id",
            "document_id",
            "chunk_index",
            name="uq_chunks_tenant_doc_idx",
        ),
        sa.ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.id"],
            ondelete="CASCADE",
            name="fk_chunks_document_tenant",
        ),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_table("document_chunks")
   
    op.drop_table("documents")

    # Leave the extension installed intentionally.
    # Other database objects may depend on it.