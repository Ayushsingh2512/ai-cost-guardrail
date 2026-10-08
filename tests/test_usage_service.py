from decimal import Decimal
from uuid import uuid4

import pytest

from app.services.models import Tenant, UsageRecord, User
from app.services.usage import (
    BudgetReservation,
    UsageService,
)

import threading

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import settings
from app.services.models import Tenant, User, UsageRecord

from app.services.usage import usage_service


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
    
def test_reserve_budget_prevents_concurrent_overspend():
    engine = create_engine(settings.database_url)

    setup_db = Session(engine)

    tenant = Tenant(
        name="Concurrency Test Tenant",
        monthly_budget=Decimal("0.001000"),
        current_spend=Decimal("0.000000"),
    )
    setup_db.add(tenant)
    setup_db.flush()

    user = User(
        tenant_id=tenant.id,
        email="concurrency-test@example.com",
    )
    setup_db.add(user)
    setup_db.commit()

    tenant_id = tenant.id
    user_id = user.id

    setup_db.close()

    first_reservation_ready = threading.Event()
    release_first_transaction = threading.Event()

    results: list[str] = []

    def first_request():
        db = Session(engine)

        try:
            usage_service.reserve_budget(
                db=db,
                tenant_id=tenant_id,
                request_id="concurrent-request-1",
                model="gemini-3-flash-preview",
                user_id=user_id,
                reserved_cost=Decimal("0.001000"),
                operation="generate",
            )

            # The reservation is flushed but not committed.
            # The tenant row lock is therefore still held.
            db.flush()

            first_reservation_ready.set()

            # Keep the transaction open so the second request
            # has to wait for the tenant row lock.
            release_first_transaction.wait(timeout=10)

            db.commit()
            results.append("first_success")

        except Exception as exc:
            db.rollback()
            results.append(f"first_error:{type(exc).__name__}")
        finally:
            db.close()

    def second_request():
        # Wait until request one has definitely acquired the lock.
        first_reservation_ready.wait(timeout=10)

        db = Session(engine)

        try:
            usage_service.reserve_budget(
                db=db,
                tenant_id=tenant_id,
                request_id="concurrent-request-2",
                model="gemini-3-flash-preview",
                user_id=user_id,
                reserved_cost=Decimal("0.001000"),
                operation="generate",
            )

            db.commit()
            results.append("second_success")

        except ValueError:
            db.rollback()
            results.append("second_budget_rejected")

        except Exception as exc:
            db.rollback()
            results.append(f"second_error:{type(exc).__name__}")

        finally:
            db.close()

    first_thread = threading.Thread(target=first_request)
    second_thread = threading.Thread(target=second_request)

    first_thread.start()

    # Ensure request one has acquired the row lock before
    # request two attempts its reservation.
    assert first_reservation_ready.wait(timeout=10)

    second_thread.start()

    # Request two is now blocked on SELECT ... FOR UPDATE.
    release_first_transaction.set()

    first_thread.join(timeout=10)
    second_thread.join(timeout=10)

    try:
        assert "first_success" in results
        assert "second_budget_rejected" in results
        assert "second_success" not in results

        verify_db = Session(engine)

        try:
            persisted_tenant = (
                verify_db.query(Tenant)
                .filter(Tenant.id == tenant_id)
                .one()
            )

            usage_records = (
                verify_db.query(UsageRecord)
                .filter(UsageRecord.tenant_id == tenant_id)
                .all()
            )

            assert persisted_tenant.current_spend == Decimal("0.001000")
            assert len(usage_records) == 1
            assert usage_records[0].request_id == "concurrent-request-1"

        finally:
            verify_db.close()

    finally:
        cleanup_db = Session(engine)

        try:
            cleanup_db.query(UsageRecord).filter(
                UsageRecord.tenant_id == tenant_id
            ).delete(synchronize_session=False)

            cleanup_db.query(User).filter(
                User.id == user_id
            ).delete(synchronize_session=False)

            cleanup_db.query(Tenant).filter(
                Tenant.id == tenant_id
            ).delete(synchronize_session=False)

            cleanup_db.commit()

        finally:
            cleanup_db.close()
            engine.dispose()