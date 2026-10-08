from decimal import Decimal

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.services.database import get_db
from app.services.models import Tenant


router = APIRouter(prefix="/tenants", tags=["Tenants"])


@router.post("/")
def create_tenant(
    name: str = Query(..., min_length=1, max_length=255),
    monthly_budget: Decimal = Query(
        Decimal("100.000000"),
        gt=Decimal("0"),
    ),
    db: Session = Depends(get_db),
):
    tenant = Tenant(
        name=name,
        monthly_budget=monthly_budget,
        current_spend=Decimal("0.000000"),
    )

    db.add(tenant)
    db.commit()
    db.refresh(tenant)

    return {
        "id": tenant.id,
        "name": tenant.name,
        "monthly_budget": tenant.monthly_budget,
    }