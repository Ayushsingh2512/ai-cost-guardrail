from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google import genai

from app.core.security import verify_access_token
from app.schemas.chat import ChatRequest
from app.schemas.rag import RAGRequest
from app.services.guardrail import RateLimitExceeded, guardrail_service
from app.services.redis_client import get_redis_client


security = HTTPBearer()

RAG_MODEL = "gemini-3-flash-preview"
RAG_MAX_OUTPUT_TOKENS = 2000


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


def _check_guardrails(
    *,
    current_user: dict,
    redis_client,
    model: str,
    max_tokens: int,
) -> None:
    try:
        guardrail_service.check_token_limit(max_tokens)

        guardrail_service.check_model_policy(model)

        guardrail_service.check_rate_limit(
            redis_client,
            int(current_user["tenant_id"]),
        )

    except RateLimitExceeded as e:
        raise HTTPException(
            status_code=429,
            detail=str(e),
        )

    except RuntimeError as e:
        raise HTTPException(
            status_code=503,
            detail=str(e),
        )

    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=str(e),
        )


def enforce_guardrails(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    redis_client=Depends(get_redis_client),
) -> ChatRequest:
    _check_guardrails(
        current_user=current_user,
        redis_client=redis_client,
        model=request.model,
        max_tokens=request.max_tokens,
    )

    return request


def enforce_rag_guardrails(
    request: RAGRequest,
    current_user: dict = Depends(get_current_user),
    redis_client=Depends(get_redis_client),
) -> RAGRequest:
    _check_guardrails(
        current_user=current_user,
        redis_client=redis_client,
        model=RAG_MODEL,
        max_tokens=RAG_MAX_OUTPUT_TOKENS,
    )

    return request