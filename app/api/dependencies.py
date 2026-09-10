from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.orm import Session
from app.services.models import Tenant
from google import genai

from app.core.security import verify_access_token
from app.services.guardrail import guardrail_service
from app.services.database import get_db
from app.schemas.chat import ChatRequest


security = HTTPBearer()





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
def enforce_guardrails(
    request: ChatRequest,
    current_user: dict = Depends(get_current_user),
    db: Session = Depends(get_db),
    
) -> ChatRequest:
    try:
        guardrail_service.check_token_limit(request.max_tokens)
        guardrail_service.check_model_policy(request.model)
        guardrail_service.check_and_reserve_budget(db, int(current_user["tenant_id"]), request.max_tokens)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return request