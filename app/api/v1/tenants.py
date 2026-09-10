from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.services.database import get_db
from app.services.models import Tenant

router = APIRouter(prefix="/tenants", tags=["Tenants"])

@router.post('/')
def create_tenant(name: str, monthly_budget: float = 100.0, db: Session = Depends(get_db)):
    tenant = Tenant(name=name, monthly_budget=monthly_budget,current_spend=0.0)
    db.add(tenant)
    db.commit()
    db.refresh(tenant)
    return {"id": tenant.id, "name": tenant.name, "monthly_budget": tenant.monthly_budget}