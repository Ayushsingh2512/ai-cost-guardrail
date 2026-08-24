from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import settings


ALGORITHM = "HS256"
TOKEN_EXPIRE_MINUTES = 60


def create_access_token(tenant_id: str, user_id: str) -> str:
    """Create a JWT containing tenant and user identity."""

    expire = datetime.now(timezone.utc) + timedelta(
        minutes=TOKEN_EXPIRE_MINUTES
    )

    payload = {
        "tenant_id": tenant_id,
        "user_id": user_id,
        "exp": expire,
    }

    return jwt.encode(
        payload,
        settings.jwt_secret,
        algorithm=ALGORITHM,
    )


def verify_access_token(token: str) -> dict:
    """Verify a JWT and return its payload."""

    return jwt.decode(
        token,
        settings.jwt_secret,
        algorithms=[ALGORITHM],
    )