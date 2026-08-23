from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    app_name: str = "AI Cost Guardrail"
    environment: str = "development"
    
    gemini_api_key: str
    redis_url: str = "redis://localhost:6379"
    jwt_secret: str
    database_url: str = "postgresql://postgres:postgres@localhost:5432/ai_guardrail"

    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,   
    )
settings = Settings()
    