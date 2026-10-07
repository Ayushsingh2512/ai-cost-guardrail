from decimal import Decimal
from uuid import uuid4

import pytest
from sqlalchemy.exc import IntegrityError

from app.services.models import Tenant, User, UsageRecord


def create_test_tenant_and_user(db):
    tenant = Tenant(
        name=f"Usage Record Test Tenant {uuid4()}",
        monthly_budget=100.0,
        current_spend=0.0,
    )

    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email=f"usage-record-{uuid4()}@example.com",
    )

    db.add(user)
    db.flush()

    return tenant, user


def test_usage_record_allows_multiple_operations_per_request(db):
    tenant, user = create_test_tenant_and_user(db)

    request_id = f"rag-{uuid4()}"

    embed_usage = UsageRecord(
        request_id=request_id,
        operation="embed",
        tenant_id=tenant.id,
        user_id=user.id,
        model="gemini-embedding-001",
        input_tokens=10,
        output_tokens=0,
        thinking_tokens=0,
        total_tokens=10,
        reserved_cost=Decimal("0.010000"),
        actual_cost=Decimal("0.010000"),
        status="completed",
    )

    generate_usage = UsageRecord(
        request_id=request_id,
        operation="generate",
        tenant_id=tenant.id,
        user_id=user.id,
        model="gemini-3-flash-preview",
        input_tokens=10,
        output_tokens=20,
        thinking_tokens=0,
        total_tokens=30,
        reserved_cost=Decimal("0.020000"),
        actual_cost=Decimal("0.015000"),
        status="completed",
    )

    db.add_all([embed_usage, generate_usage])
    db.commit()

    assert embed_usage.id is not None
    assert generate_usage.id is not None


def test_create_usage_record(db):
    tenant = Tenant(
        name="Usage Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )

    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="usage-test@example.com",
    )

    db.add(user)
    db.flush()

    request_id = f"test-request-{uuid4()}"

    usage = UsageRecord(
        request_id=request_id,
        tenant_id=tenant.id,
        user_id=user.id,
        model="gemini-3-flash-preview",
        operation="generate",
        input_tokens=100,
        output_tokens=200,
        thinking_tokens=0,
        total_tokens=300,
        reserved_cost=Decimal("0.001000"),
        actual_cost=Decimal("0.000300"),
        status="completed",
    )

    db.add(usage)
    db.commit()
    db.refresh(usage)

    assert usage.id is not None
    assert usage.request_id == request_id
    assert usage.operation == "generate"
    assert usage.tenant_id == tenant.id
    assert usage.user_id == user.id
    assert usage.total_tokens == 300
    assert usage.reserved_cost == Decimal("0.001000")
    assert usage.actual_cost == Decimal("0.000300")
    assert usage.status == "completed"


def test_usage_record_rejects_duplicate_request_and_operation(db):
    tenant = Tenant(
        name="Duplicate Request Test Tenant",
        monthly_budget=100.0,
        current_spend=0.0,
    )

    db.add(tenant)
    db.flush()

    user = User(
        tenant_id=tenant.id,
        email="duplicate-test@example.com",
    )

    db.add(user)
    db.flush()

    request_id = f"duplicate-request-{uuid4()}"

    first_usage = UsageRecord(
        request_id=request_id,
        tenant_id=tenant.id,
        user_id=user.id,
        model="gemini-3-flash-preview",
        operation="generate",
        input_tokens=100,
        output_tokens=200,
        thinking_tokens=0,
        total_tokens=300,
        reserved_cost=Decimal("0.001000"),
        actual_cost=Decimal("0.000300"),
        status="completed",
    )

    db.add(first_usage)
    db.commit()

    duplicate_usage = UsageRecord(
        request_id=request_id,
        tenant_id=tenant.id,
        user_id=user.id,
        model="gemini-3-flash-preview",
        operation="generate",
        input_tokens=100,
        output_tokens=200,
        thinking_tokens=0,
        total_tokens=300,
        reserved_cost=Decimal("0.001000"),
        actual_cost=Decimal("0.000300"),
        status="completed",
    )

    db.add(duplicate_usage)

    with pytest.raises(IntegrityError):
        db.commit()

    db.rollback()