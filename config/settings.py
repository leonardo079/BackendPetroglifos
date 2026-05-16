"""Configuración centralizada via pydantic-settings + .env."""
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # Gemini
    gemini_api_key: str
    gemini_model: str = "gemini-1.5-flash"
    embedding_model: str = "text-embedding-004"
    # PostgreSQL + pgvector
    database_url: str
    # RAG
    rag_top_k: int = 5
    rag_min_similarity: float = 0.55
    confidence_threshold: float = 0.70
    # Object storage
    storage_bucket: str = "petroglifos"

    class Config:
        env_file = ".env"

settings = Settings()
