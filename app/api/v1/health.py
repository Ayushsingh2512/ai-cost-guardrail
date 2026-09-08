from fastapi import APIRouter, HTTPException, Depends
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.redis_client import get_redis_client, ping_redis
from app.services.database import get_db

router = APIRouter(prefix="/health", tags=["Health"])


@router.get("/redis")
def health_redis():
    if ping_redis():
        return {"redis": "connected"}
    raise HTTPException(status_code=503, detail="Redis unavailable")


@router.get("/db")
def health_db(db: Session = Depends(get_db)):
    try:
        db.execute(text("SELECT 1"))
        return {"database": "connected"}
    except Exception:
        raise HTTPException(status_code=503, detail="Database unavailable")


def test_set_get():
    client = get_redis_client()
    client.set("test_key", "hello", ex=10)
    return client.get("test_key")