from fastapi import FastAPI
from dotenv import load_dotenv
from app.api.v1 import chat


load_dotenv()

app = FastAPI(title="AI Cost Guardrail")

# Plug in the power strip
app.include_router(chat.router)

@app.get("/")
def home():
    return {"message": "AI Cost Guardrail is running"}