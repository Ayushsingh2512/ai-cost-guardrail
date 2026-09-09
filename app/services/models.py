from datetime import datetime
from sqlalchemy import ForeignKey, String, Float, DateTime, func
from sqlalchemy.orm import Mapped, mapped_column, relationship
from app.services.database import Base


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String)
    monthly_budget: Mapped[float] = mapped_column(Float, default=100.0)
    current_spend: Mapped[float] = mapped_column(Float,default=0.0)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    
    users: Mapped[list["User"]] = relationship(back_populates="tenant")
    
class User(Base):
    __tablename__ = "users"
   
    id: Mapped[int] = mapped_column(primary_key=True)
    tenant_id: Mapped[int] = mapped_column(ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(String)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now()) 
    tenant: Mapped["Tenant"] = relationship(back_populates = "users")