from fastapi import APIRouter, Depends, HTTPException
from app.services.guardrail import guardrail_service
from app.api.dependencies import (
    enforce_guardrails,
    get_current_user,
    get_genai_client,
)
from sqlalchemy.orm import Session
from app.services.models import Tenant
from app.schemas.chat import ChatRequest
from app.services.database import get_db


router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest = Depends(enforce_guardrails),
    current_user: dict = Depends(get_current_user),
    client=Depends(get_genai_client),
    db: Session = Depends(get_db),
):
    user_id = current_user["user_id"]
    tenant_id = int(current_user["tenant_id"])
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    if tenant is None:
        raise HTTPException(status_code=404, detail=f"Tenant {tenant_id} not found")

    try:
        response = await client.aio.models.generate_content(
            model=request.model,
            contents=request.message,
            config={"max_output_tokens": request.max_tokens},
        )

        # Look at the receipt from Gemini
        actual_tokens = response.usage_metadata.total_token_count

        # Refund unused tokens
        if actual_tokens < request.max_tokens:
            unused_tokens = request.max_tokens - actual_tokens
            refund_amount = (
                unused_tokens / 1000
            ) * guardrail_service.COST_PER_1000_TOKENS

            tenant.current_spend -= refund_amount
            db.commit()
            

        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "received_message": request.message,
            "actual_tokens_used": actual_tokens,
            "ai_response": response.text,
            "total_spend": round(tenant.current_spend, 6),
        }

    except Exception as e:
        # Failure refund
        full_refund = (
            request.max_tokens / 1000
        ) * guardrail_service.COST_PER_1000_TOKENS

        tenant.current_spend -= full_refund
        db.commit()

        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM Error: {str(e)}",
        )