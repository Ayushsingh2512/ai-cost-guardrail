from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str = "bay harbour butcher"
    model: str = "gemini-3-flash-preview"
    max_tokens: int = 500