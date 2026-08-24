from fastapi import APIRouter, Depends, HTTPException

from app.api.dependencies import (
    COST_PER_1000_TOKENS,
    enforce_budget,
    get_current_user,
    get_genai_client,
    user_spend,
)
from app.schemas.chat import ChatRequest


router = APIRouter(prefix="/api/v1", tags=["Chat"])


@router.post("/chat")
async def chat(
    request: ChatRequest = Depends(enforce_budget),
    current_user: dict = Depends(get_current_user),
    client=Depends(get_genai_client),
):
    user_id = current_user["user_id"]
    tenant_id = current_user["tenant_id"]

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
            ) * COST_PER_1000_TOKENS

            user_spend[user_id] -= refund_amount

        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
            "received_message": request.message,
            "actual_tokens_used": actual_tokens,
            "ai_response": response.text,
            "total_spend": round(user_spend[user_id], 6),
        }

    except Exception as e:
        # Failure refund
        full_refund = (
            request.max_tokens / 1000
        ) * COST_PER_1000_TOKENS

        user_spend[user_id] -= full_refund

        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM Error: {str(e)}",
        )