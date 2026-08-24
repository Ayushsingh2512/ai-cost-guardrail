from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai

from app.core.security import verify_access_token
from app.schemas.chat import ChatRequest


security = HTTPBearer()


# Global state (We will move this to Redis in Sprint 4)
ALLOWED_MODELS = [
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
]

DAILY_BUDGET = 1.0
COST_PER_1000_TOKENS = 0.001

user_spend: dict[str, float] = {}


def get_genai_client():
    return genai.Client()


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """Extract and verify the JWT from the Authorization header."""

    try:
        payload = verify_access_token(credentials.credentials)

        tenant_id = payload.get("tenant_id")
        user_id = payload.get("user_id")

        if not tenant_id or not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token payload",
            )

        return {
            "tenant_id": tenant_id,
            "user_id": user_id,
        }

    except HTTPException:
        raise

    except Exception:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
        )


def enforce_token_limit(
    request: ChatRequest,
) -> ChatRequest:

    if request.max_tokens > 2000:
        raise HTTPException(
            status_code=400,
            detail=(
                f"max_tokens ({request.max_tokens}) "
                "exceeds allowed limit of 2000"
            ),
        )

    return request


def enforce_model_policy(
    request: ChatRequest = Depends(enforce_token_limit),
) -> ChatRequest:

    if request.model not in ALLOWED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{request.model}' is not allowed. "
                f"Allowed models: {ALLOWED_MODELS}"
            ),
        )

    return request


def estimate_cost(max_tokens: int) -> float:
    return (max_tokens / 1000) * COST_PER_1000_TOKENS


def enforce_budget(
    request: ChatRequest = Depends(enforce_model_policy),
    current_user: dict = Depends(get_current_user),
) -> ChatRequest:

    user_id = current_user["user_id"]

    cost = estimate_cost(request.max_tokens)
    current_spend = user_spend.get(user_id, 0.0)

    if current_spend + cost > DAILY_BUDGET:
        raise HTTPException(
            status_code=400,
            detail=(
                "Request would exceed daily budget. "
                f"Spent so far: ${current_spend:.4f}, "
                f"Limit: ${DAILY_BUDGET:.4f}"
            ),
        )

    # Reserve worst-case cost upfront
    user_spend[user_id] = current_spend + cost

    return request