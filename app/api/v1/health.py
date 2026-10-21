from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.services.database import get_db
from app.services.redis_client import ping_redis

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
