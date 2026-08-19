from fastapi import APIRouter, Depends, HTTPException
from app.schemas.chat import ChatRequest
from app.api.dependencies import (
    enforce_budget, 
    get_genai_client, 
    user_spend, 
    COST_PER_1000_TOKENS
)

router = APIRouter(prefix="/api/v1", tags=["Chat"])

@router.post("/chat")
async def chat(
    request: ChatRequest = Depends(enforce_budget),
    client = Depends(get_genai_client),
):
    user_id = "demo_user"
    
    try:
        response = await client.aio.models.generate_content(
            model=request.model,
            contents=request.message,
            config={"max_output_tokens": request.max_tokens},
        )
        
        # 1. Look at the receipt from Gemini
        actual_tokens = response.usage_metadata.total_token_count
        
        # 2. THE REFUND: Calculate the change owed
        if actual_tokens < request.max_tokens:
            unused_tokens = request.max_tokens - actual_tokens
            refund_amount = (unused_tokens / 1000) * COST_PER_1000_TOKENS
            user_spend[user_id] -= refund_amount
            
        return {
            "received_message": request.message,
            "actual_tokens_used": actual_tokens,
            "ai_response": response.text,
            "total_spend": round(user_spend[user_id], 6),
        }

    except Exception as e:
        # THE FAILURE REFUND: If Gemini crashes, give them all their money back
        full_refund = (request.max_tokens / 1000) * COST_PER_1000_TOKENS
        user_spend[user_id] -= full_refund
        
        raise HTTPException(
            status_code=502,
            detail=f"Upstream LLM Error: {str(e)}"
        )