from dataclasses import dataclass
from decimal import Decimal
from typing import Literal

from sqlalchemy.orm import Session

from app.services.models import Tenant, UsageRecord


UsageOperation = Literal["embed", "generate"]


@dataclass(frozen=True)
class BudgetReservation:
    operation: UsageOperation
    model: str
    reserved_cost: Decimal


class UsageService:
    def reserve_budget(
        self,
        db: Session,
        tenant_id: int,
        request_id: str,
        model: str,
        user_id: int,
        reserved_cost: Decimal,
        operation: UsageOperation = "generate",
    ) -> UsageRecord:
        """
        Reserve the estimated maximum cost for a single operation.

        The reservation temporarily increases tenant.current_spend.
        The actual cost will be settled after the provider response.
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

        tenant.current_spend = current_spend + reserved_cost

        usage = UsageRecord(
            request_id=request_id,
            operation=operation,
            tenant_id=tenant_id,
            user_id=user_id,
            model=model,
            input_tokens=0,
            output_tokens=0,
            thinking_tokens=0,
            total_tokens=0,
            reserved_cost=reserved_cost,
            actual_cost=Decimal("0"),
            status="reserved",
        )

        db.add(usage)
        db.flush()

        return usage

    def reserve_budget_batch(
        self,
        db: Session,
        tenant_id: int,
        request_id: str,
        user_id: int,
        reservations: list[BudgetReservation],
    ) -> list[UsageRecord]:
        """
        Reserve the combined estimated maximum cost for multiple
        provider operations belonging to one request.

        The tenant row is locked and the complete reservation is checked
        against the budget before any usage records are created.
        """

        if not reservations:
            raise ValueError("At least one budget reservation is required")

        operations = [reservation.operation for reservation in reservations]

        if len(operations) != len(set(operations)):
            raise ValueError("Duplicate operations are not allowed in one request")

        total_reserved = sum(
            (reservation.reserved_cost for reservation in reservations),
            Decimal("0"),
        )

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

        if current_spend + total_reserved > monthly_budget:
            raise ValueError(
                f"Request would exceed budget. "
                f"Spent so far: ${current_spend:.6f}, "
                f"Reserved: ${total_reserved:.6f}, "
                f"Limit: ${monthly_budget:.6f}"
            )

        tenant.current_spend = current_spend + total_reserved

        usage_records: list[UsageRecord] = []

        for reservation in reservations:
            usage = UsageRecord(
                request_id=request_id,
                operation=reservation.operation,
                tenant_id=tenant_id,
                user_id=user_id,
                model=reservation.model,
                input_tokens=0,
                output_tokens=0,
                thinking_tokens=0,
                total_tokens=0,
                reserved_cost=reservation.reserved_cost,
                actual_cost=Decimal("0"),
                status="reserved",
            )

            db.add(usage)
            usage_records.append(usage)

        db.flush()

        return usage_records

    def settle_success(
        self,
        db: Session,
        usage: UsageRecord,
        actual_cost: Decimal,
        input_tokens: int,
        thinking_tokens: int,
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

        reserved_cost = Decimal(str(usage.reserved_cost))
        refund = reserved_cost - actual_cost

        current_spend = Decimal(str(tenant.current_spend))
        tenant.current_spend = current_spend - refund

        usage.input_tokens = input_tokens
        usage.thinking_tokens = thinking_tokens
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
        Release the entire reservation after a provider failure.
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
        tenant.current_spend = current_spend - Decimal(
            str(usage.reserved_cost)
        )

        usage.actual_cost = Decimal("0")
        usage.status = "failed"

        db.flush()


usage_service = UsageService()