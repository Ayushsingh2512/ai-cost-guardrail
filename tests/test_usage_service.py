from decimal import Decimal
from uuid import uuid4

import pytest

from app.services.models import Tenant, UsageRecord, User
from app.services.usage import (
    BudgetReservation,
    UsageService,
)


def create_test_tenant_and_user(db):
    tenant = Tenant(
        name=f"Usage Service Test Tenant {uuid4()}",
        monthly_budget=1.00,
        current_spend=0.00,
    )

    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email=f"usage-service-{uuid4()}@example.com",
    )

    db.add(user)
    db.flush()

    return tenant, user


def test_reserve_budget(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    reserved_cost = Decimal("0.050000")

    usage = service.reserve_budget(
        db=db,
        tenant_id=tenant.id,
        request_id=f"request-{uuid4()}",
        model="gemini-3-flash-preview",
        user_id=user.id,
        reserved_cost=reserved_cost,
    )

    assert usage.id is not None
    assert usage.operation == "generate"
    assert usage.status == "reserved"
    assert usage.reserved_cost == reserved_cost
    assert usage.actual_cost == Decimal("0")

    db.refresh(tenant)

    assert Decimal(str(tenant.current_spend)) == Decimal("0.050000")


def test_reservation_rejected_when_budget_exceeded(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    tenant.monthly_budget = 0.05
    db.flush()

    with pytest.raises(ValueError, match="would exceed budget"):
        service.reserve_budget(
            db=db,
            tenant_id=tenant.id,
            request_id=f"request-{uuid4()}",
            model="gemini-3-flash-preview",
            user_id=user.id,
            reserved_cost=Decimal("0.050001"),
        )

    db.refresh(tenant)

    assert Decimal(str(tenant.current_spend)) == Decimal("0")


def test_successful_settlement_refunds_unused_reservation(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    reserved_cost = Decimal("0.050000")
    actual_cost = Decimal("0.030000")

    usage = service.reserve_budget(
        db=db,
        tenant_id=tenant.id,
        request_id=f"request-{uuid4()}",
        model="gemini-3-flash-preview",
        user_id=user.id,
        reserved_cost=reserved_cost,
    )

    service.settle_success(
        db=db,
        usage=usage,
        actual_cost=actual_cost,
        input_tokens=100,
        thinking_tokens=50,
        output_tokens=200,
        total_tokens=350,
    )

    db.refresh(tenant)
    db.refresh(usage)

    assert Decimal(str(tenant.current_spend)) == Decimal("0.030000")

    assert usage.status == "completed"
    assert usage.reserved_cost == reserved_cost
    assert usage.actual_cost == actual_cost
    assert usage.input_tokens == 100
    assert usage.thinking_tokens == 50
    assert usage.output_tokens == 200
    assert usage.total_tokens == 350


def test_failed_request_releases_reservation(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    reserved_cost = Decimal("0.050000")

    usage = service.reserve_budget(
        db=db,
        tenant_id=tenant.id,
        request_id=f"request-{uuid4()}",
        model="gemini-3-flash-preview",
        user_id=user.id,
        reserved_cost=reserved_cost,
    )

    service.settle_failure(
        db=db,
        usage=usage,
    )

    db.refresh(tenant)
    db.refresh(usage)

    assert Decimal(str(tenant.current_spend)) == Decimal("0")

    assert usage.status == "failed"
    assert usage.actual_cost == Decimal("0")


def test_successful_settlement_increases_spend_when_actual_cost_exceeds_reservation(
    db,
):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    reserved_cost = Decimal("0.050000")
    actual_cost = Decimal("0.070000")

    usage = service.reserve_budget(
        db=db,
        tenant_id=tenant.id,
        request_id=f"request-{uuid4()}",
        model="gemini-3-flash-preview",
        user_id=user.id,
        reserved_cost=reserved_cost,
    )

    service.settle_success(
        db=db,
        usage=usage,
        actual_cost=actual_cost,
        input_tokens=100,
        thinking_tokens=50,
        output_tokens=200,
        total_tokens=350,
    )

    db.refresh(tenant)
    db.refresh(usage)

    assert Decimal(str(tenant.current_spend)) == Decimal("0.070000")

    assert usage.status == "completed"
    assert usage.reserved_cost == reserved_cost
    assert usage.actual_cost == actual_cost
    assert usage.input_tokens == 100
    assert usage.thinking_tokens == 50
    assert usage.output_tokens == 200
    assert usage.total_tokens == 350


def test_reserve_budget_batch_creates_embed_and_generate_records(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    request_id = f"rag-{uuid4()}"

    reservations = [
        BudgetReservation(
            operation="embed",
            model="gemini-embedding-001",
            reserved_cost=Decimal("0.010000"),
        ),
        BudgetReservation(
            operation="generate",
            model="gemini-3-flash-preview",
            reserved_cost=Decimal("0.050000"),
        ),
    ]

    usage_records = service.reserve_budget_batch(
        db=db,
        tenant_id=tenant.id,
        request_id=request_id,
        user_id=user.id,
        reservations=reservations,
    )

    assert len(usage_records) == 2

    operations = {usage.operation for usage in usage_records}

    assert operations == {"embed", "generate"}

    assert Decimal(str(tenant.current_spend)) == Decimal("0.060000")

    for usage in usage_records:
        assert usage.request_id == request_id
        assert usage.status == "reserved"


def test_reserve_budget_batch_rejects_combined_budget_exceeded(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    tenant.monthly_budget = Decimal("0.05")
    tenant.current_spend = Decimal("0.00")
    db.flush()

    request_id = f"rag-{uuid4()}"

    reservations = [
        BudgetReservation(
            operation="embed",
            model="gemini-embedding-001",
            reserved_cost=Decimal("0.010000"),
        ),
        BudgetReservation(
            operation="generate",
            model="gemini-3-flash-preview",
            reserved_cost=Decimal("0.050000"),
        ),
    ]

    with pytest.raises(ValueError, match="would exceed budget"):
        service.reserve_budget_batch(
            db=db,
            tenant_id=tenant.id,
            request_id=request_id,
            user_id=user.id,
            reservations=reservations,
        )

    db.refresh(tenant)

    assert Decimal(str(tenant.current_spend)) == Decimal("0.00")

    usage_records = (
        db.query(UsageRecord)
        .filter(UsageRecord.request_id == request_id)
        .all()
    )

    assert usage_records == []


def test_reserve_budget_batch_rejects_duplicate_operation(db):
    service = UsageService()

    tenant, user = create_test_tenant_and_user(db)

    reservations = [
        BudgetReservation(
            operation="generate",
            model="gemini-3-flash-preview",
            reserved_cost=Decimal("0.010000"),
        ),
        BudgetReservation(
            operation="generate",
            model="gemini-3-flash-preview",
            reserved_cost=Decimal("0.010000"),
        ),
    ]

    with pytest.raises(
        ValueError,
        match="Duplicate operations are not allowed",
    ):
        service.reserve_budget_batch(
            db=db,
            tenant_id=tenant.id,
            request_id=f"rag-{uuid4()}",
            user_id=user.id,
            reservations=reservations,
        )

    db.refresh(tenant)

    assert Decimal(str(tenant.current_spend)) == Decimal("0.00")