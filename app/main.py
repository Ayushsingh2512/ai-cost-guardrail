from fastapi import FastAPI
from dotenv import load_dotenv

from app.api.v1 import chat
from app.core.security import create_access_token

load_dotenv()

app = FastAPI(title="AI Cost Guardrail")

app.include_router(chat.router)


@app.get("/")
def home():
    return {"message": "AI Cost Guardrail is running"}


@app.post("/token")
def generate_test_token(
    tenant_id: str,
    user_id: str,
):
    return {
        "access_token": create_access_token(
            tenant_id=tenant_id,
            user_id=user_id,
        ),
        "token_type": "bearer",
    }
from app.api.v1 import chat, health

app.include_router(chat.router)
app.include_router(health.router)