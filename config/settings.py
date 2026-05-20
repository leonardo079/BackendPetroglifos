"""Configuración centralizada via pydantic-settings + .env."""
from __future__ import annotations
from functools import lru_cache
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_model_lite: str = "gemini-1.5-flash-8b"
    embedding_model: str = "text-embedding-004"

    # PostgreSQL + pgvector
    database_url: str = "postgresql+asyncpg://petro:petro@localhost:5432/petroglifos"
    database_url_sync: str = "postgresql+psycopg2://petro:petro@localhost:5432/petroglifos"

    # RAG
    rag_top_k: int = 5
    rag_min_similarity: float = 0.55
    confidence_threshold: float = 0.70

    # Object storage
    storage_bucket: str = "petroglifos"
    storage_endpoint: str = "http://localhost:9000"
    storage_access_key: str = "minioadmin"
    storage_secret_key: str = "minioadmin"

    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/1"

    # Reconstruction API (ProyectoPetroglifosBack)
    # En Docker: http://reconstruction:8001  |  Local: http://localhost:8001
    reconstruction_api_base_url: str = "http://localhost:8001"
    gan_api_url: str = "http://localhost:8001/reconstruct"
    gan_api_key: str = ""
    gan_mock_mode: bool = True

    # Telegram
    telegram_bot_token: str = ""

    # App
    env: str = "development"
    log_level: str = "INFO"
    max_record_sheet_minutes: int = 45
    # URL base de la API FastAPI — usada por el bot de Telegram
    # En Docker: http://api:8000  |  Local: http://localhost:8000
    api_base_url: str = "http://localhost:8000"


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()