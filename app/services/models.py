from datetime import datetime
from decimal import Decimal
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.services.database import Base
from sqlalchemy import ForeignKey, String, DateTime, Numeric, func


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

    id: Mapped[int] = mapped_column(primary_key=True)

    request_id: Mapped[str] = mapped_column(
        String,
        unique=True,
        nullable=False,
        index=True,
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