from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from uuid import uuid4

from app.api.dependencies import (
    enforce_guardrails,
    get_current_user,
    get_genai_client,
)
from app.schemas.chat import ChatRequest
from app.services.circuit_breaker import circuit_breaker
from app.services.database import get_db
from app.services.cost_engine import cost_engine
from app.services.models import Tenant
from app.services.usage import usage_service


router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest = Depends(enforce_guardrails),
    current_user: dict = Depends(get_current_user),
    client=Depends(get_genai_client),
    db: Session = Depends(get_db),
):
    tenant_id = int(current_user["tenant_id"])
    user_id = int(current_user["user_id"])

    # ─────────────────────────────────────
    # Find tenant
    # ─────────────────────────────────────

    tenant = (
        db.query(Tenant)
        .filter(Tenant.id == tenant_id)
        .first()
    )

    if tenant is None:
        raise HTTPException(
            status_code=404,
            detail=f"Tenant {tenant_id} not found",
        )

    # ─────────────────────────────────────
    # Generate request ID
    # ─────────────────────────────────────

    request_id = str(uuid4())

    # ─────────────────────────────────────
    # Calculate reservation
    # ─────────────────────────────────────
        # Calculate reservation

    token_count = await client.aio.models.count_tokens(
        model=request.model,
        contents=request.message,
    )

    input_tokens_estimate = token_count.total_tokens or 0

    reserved_cost = cost_engine.estimate_reservation(
        model=request.model,
        input_tokens=input_tokens_estimate,
        max_output_tokens=request.max_tokens,
    )

    # ─────────────────────────────────────
    # Reserve budget
    # ─────────────────────────────────────

    try:
        usage = usage_service.reserve_budget(
            db=db,
            tenant_id=tenant_id,
            request_id=request_id,
            model=request.model,
            user_id=user_id,
            reserved_cost=reserved_cost,
        )

        db.commit()

    except ValueError as e:
        db.rollback()

        raise HTTPException(
            status_code=400,
            detail=str(e),
        )

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="Unable to reserve request budget",
        )

    # ─────────────────────────────────────
    # Circuit breaker
    # ─────────────────────────────────────

    if not circuit_breaker.allow_request():
        usage_service.settle_failure(
            db=db,
            usage=usage,
        )

        db.commit()

        raise HTTPException(
            status_code=503,
            detail=(
                "LLM provider is currently unavailable "
                "(circuit open). Try again shortly."
            ),
        )

    # ─────────────────────────────────────
    # Call upstream LLM
    # ─────────────────────────────────────

    try:
        response = await client.aio.models.generate_content(
            model=request.model,
            contents=request.message,
            config={
                "max_output_tokens": request.max_tokens,
            },
        )

    except Exception as e:
        circuit_breaker.record_failure()

        try:
            usage_service.settle_failure(
                db=db,
                usage=usage,
            )
            db.commit()

        except Exception:
            db.rollback()

            raise HTTPException(
                status_code=500,
                detail="LLM failed and usage settlement also failed",
            )

        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM Error: {str(e)}",
        )

    # Gemini successfully responded.
    circuit_breaker.record_success()

    # ─────────────────────────────────────
    # Extract actual usage
    # ─────────────────────────────────────

    usage_metadata = response.usage_metadata

    input_tokens = usage_metadata.prompt_token_count or 0
    output_tokens = usage_metadata.candidates_token_count or 0
    total_tokens = usage_metadata.total_token_count or (
        input_tokens + output_tokens
    )

    # ─────────────────────────────────────
    # Calculate actual cost
    # ─────────────────────────────────────

    actual_cost = cost_engine.calculate_actual_cost(
    model=request.model,
    input_tokens=input_tokens,
    output_tokens=output_tokens,
    )
    

    # ─────────────────────────────────────
    # Settle reservation
    # ─────────────────────────────────────

    try:
        usage_service.settle_success(
            db=db,
            usage=usage,
            actual_cost=actual_cost,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )

        db.commit()
        db.refresh(usage)
        db.refresh(tenant)

    except Exception:
        db.rollback()

        raise HTTPException(
            status_code=500,
            detail="LLM succeeded but usage settlement failed",
        )

    # ─────────────────────────────────────
    # Return response
    # ─────────────────────────────────────

    return {
        "request_id": request_id,
        "tenant_id": tenant_id,
        "user_id": user_id,
        "received_message": request.message,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "reserved_cost": float(usage.reserved_cost),
        "actual_cost": float(usage.actual_cost),
        "status": usage.status,
        "ai_response": response.text,
        "total_spend": round(tenant.current_spend, 6),
    }