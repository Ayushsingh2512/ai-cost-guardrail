# app/api/v1/health.py

from fastapi import APIRouter, HTTPException

from app.services.redis_client import get_redis_client, ping_redis

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/redis")
def health_redis():
    if ping_redis():
        return {"redis": "connected"}
    raise HTTPException(status_code=503, detail="Redis unavailable")


def test_set_get():
    client = get_redis_client()
    client.set("test_key", "hello", ex=10)
    return client.get("test_key")