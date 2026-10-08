from fastapi import FastAPI

from app.api.v1 import chat, health, rag, tenants
from app.core.config import Settings, settings
from app.core.security import create_access_token


def create_app(app_settings: Settings) -> FastAPI:
    app = FastAPI(title=app_settings.app_name)

    app.include_router(chat.router)
    app.include_router(health.router)
    app.include_router(rag.router)

    @app.get("/")
    def home():
        return {"message": "AI Cost Guardrail is running"}

    if app_settings.environment == "development":
        app.include_router(tenants.router)

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

    return app


app = create_app(settings)