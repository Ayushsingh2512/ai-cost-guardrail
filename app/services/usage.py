from decimal import Decimal

from sqlalchemy.orm import Session

from app.services.models import Tenant, UsageRecord


class UsageService:
    def reserve_budget(
        self,
        db: Session,
        tenant_id: int,
        request_id: str,
        model: str,
        user_id: int,
        reserved_cost: Decimal,
    ) -> UsageRecord:
        """
        Reserve the estimated maximum cost for a request.

        The reservation temporarily increases tenant.current_spend.
        The actual cost will be settled after the LLM response.
        """

        tenant = (
            db.query(Tenant)
            .filter(Tenant.id == tenant_id)
            .with_for_update()
            .first()
        )

        if tenant is None:
            raise ValueError(f"Tenant {tenant_id} not found")

        current_spend = Decimal(str(tenant.current_spend))
        monthly_budget = Decimal(str(tenant.monthly_budget))

        if current_spend + reserved_cost > monthly_budget:
            raise ValueError(
                f"Request would exceed budget. "
                f"Spent so far: ${current_spend:.6f}, "
                f"Reserved: ${reserved_cost:.6f}, "
                f"Limit: ${monthly_budget:.6f}"
            )

        tenant.current_spend = float(current_spend + reserved_cost)

        usage = UsageRecord(
            request_id=request_id,
            tenant_id=tenant_id,
            user_id=user_id,
            model=model,
            input_tokens=0,
            output_tokens=0,
            total_tokens=0,
            reserved_cost=reserved_cost,
            actual_cost=Decimal("0"),
            status="reserved",
        )

        db.add(usage)
        db.flush()

        return usage

    def settle_success(
        self,
        db: Session,
        usage: UsageRecord,
        actual_cost: Decimal,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
    ) -> None:
        """
        Replace the temporary reservation with the actual cost.
        """

        tenant = (
            db.query(Tenant)
            .filter(Tenant.id == usage.tenant_id)
            .with_for_update()
            .first()
        )

        if tenant is None:
            raise ValueError(f"Tenant {usage.tenant_id} not found")

        reserved_cost = usage.reserved_cost

        refund = reserved_cost - actual_cost

        current_spend = Decimal(str(tenant.current_spend))

        tenant.current_spend = float(current_spend - refund)

        usage.input_tokens = input_tokens
        usage.output_tokens = output_tokens
        usage.total_tokens = total_tokens
        usage.actual_cost = actual_cost
        usage.status = "completed"

        db.flush()

    def settle_failure(
        self,
        db: Session,
        usage: UsageRecord,
    ) -> None:
        """
        Release the entire reservation after an LLM failure.
        """

        tenant = (
            db.query(Tenant)
            .filter(Tenant.id == usage.tenant_id)
            .with_for_update()
            .first()
        )

        if tenant is None:
            raise ValueError(f"Tenant {usage.tenant_id} not found")

        current_spend = Decimal(str(tenant.current_spend))

        tenant.current_spend = float(
            current_spend - usage.reserved_cost
        )

        usage.actual_cost = Decimal("0")
        usage.status = "failed"

        db.flush()


usage_service = UsageService()