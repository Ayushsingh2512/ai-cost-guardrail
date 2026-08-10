from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()





class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-2.0-flash"
    max_tokens: int = 500


def enforce_token_limit(request: ChatRequest) -> ChatRequest:
    if request.max_tokens > 2000:
        raise HTTPException(
            status_code=400,
            detail=f"max_tokens ({request.max_tokens}) exceeds allowed limit of 2000",
        )

    return request  # ✅ Now executes for all valid requests


@app.get("/")
def home():
    return {"message": "AI Cost Guardrail is running"}


@app.post("/chat")
def chat(request: ChatRequest = Depends(enforce_token_limit)):
    return {
        "received_message": request.message,
        "model_requested": request.model,
        "max_tokens_requested": request.max_tokens,
    } 