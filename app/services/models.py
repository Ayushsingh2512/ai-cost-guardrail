import uuid
from datetime import datetime
from decimal import Decimal

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    DateTime,
    ForeignKey,
    ForeignKeyConstraint,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.services.database import Base

EMBEDDING_DIMS = 768


class Document(Base):
    __tablename__ = "documents"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "id",
            name="uq_documents_tenant_id_id",
        ),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )

    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )

    source: Mapped[str | None] = mapped_column(Text)

    created_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        nullable=False,
    )


class DocumentChunkRecord(Base):
    __tablename__ = "document_chunks"

    __table_args__ = (
        UniqueConstraint(
            "tenant_id",
            "document_id",
            "chunk_index",
            name="uq_chunks_tenant_doc_idx",
        ),
        ForeignKeyConstraint(
            ["tenant_id", "document_id"],
            ["documents.tenant_id", "documents.id"],
            ondelete="CASCADE",
            name="fk_chunks_document_tenant",
        ),
    )

    id: Mapped[int] = mapped_column(
        primary_key=True,
        autoincrement=True,
    )

    tenant_id: Mapped[int] = mapped_column(
        nullable=False,
    )

    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        nullable=False,
    )

    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )

    page: Mapped[int | None] = mapped_column(Integer)

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )

    metadata_: Mapped[dict] = mapped_column(
        "metadata",
        JSONB,
        nullable=False,
        server_default="{}",
    )

    embedding = mapped_column(
        Vector(EMBEDDING_DIMS),
        nullable=False,
    )

    fingerprint: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    monthly_budget: Mapped[Decimal] = mapped_column(
    Numeric(12, 6),
    default=Decimal("100.000000"),
    )

    current_spend: Mapped[Decimal] = mapped_column(
    Numeric(12, 6),
    default=Decimal("0.000000"),
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    users: Mapped[list["User"]] = relationship(back_populates="tenant")
    
class User(Base):
    __tablename__ = "users"
   
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now()) 
    tenant: Mapped["Tenant"] = relationship(back_populates = "users")
    
class UsageRecord(Base):
    __tablename__ = "usage_records"
    
    __table_args__ = (
        UniqueConstraint(
            "request_id",
            "operation",
            name="uq_usage_records_request_operation",
        ),
    )

    id: Mapped[int] = mapped_column(primary_key=True)

    request_id: Mapped[str] = mapped_column(
        String,
        nullable=False,
        index=True,
    )

    operation: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    tenant_id: Mapped[int] = mapped_column(
        ForeignKey("tenants.id"),
        nullable=False,
        index=True,
    )

    user_id: Mapped[int] = mapped_column(
        ForeignKey("users.id"),
        nullable=False,
        index=True,
    )

    model: Mapped[str] = mapped_column(
        String,
        nullable=False,
    )

    input_tokens: Mapped[int] = mapped_column(
        nullable=False,
    )

    output_tokens: Mapped[int] = mapped_column(
        nullable=False,
    )

    thinking_tokens: Mapped[int] = mapped_column(
        nullable=False,
    )

    total_tokens: Mapped[int] = mapped_column(
        nullable=False,
    )

    reserved_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        nullable=False,
    )

    actual_cost: Mapped[Decimal] = mapped_column(
        Numeric(12, 6),
        nullable=False,
    )

    status: Mapped[str] = mapped_column(
        String(20),
        nullable=False,
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime,
        server_default=func.now(),
        nullable=False,
    )