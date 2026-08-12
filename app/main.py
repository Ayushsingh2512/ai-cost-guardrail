from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from google import genai
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()





class ChatRequest(BaseModel):
    message: str
    model: str = "gemini-3-flash-preview"
    max_tokens: int = 500
def get_genai_client():
    return genai.Client()

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
async def chat(
    request: ChatRequest = Depends(enforce_token_limit),
    client = Depends(get_genai_client)
):
    response = await client.aio.models.generate_content(
        model=request.model,
        contents=request.message,
    )
    return {
        "received_message": request.message,
        "model_used": request.model,
        "ai_response": response.text,
    } 