"""Configuración centralizada via pydantic-settings + .env."""
from __future__ import annotations
from functools import lru_cache
from urllib.parse import urlsplit
from pydantic import model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Gemini
    gemini_api_key: str = ""
    gemini_model: str = "gemini-1.5-flash"
    gemini_model_lite: str = "gemini-1.5-flash"
    embedding_model: str = "gemini-embedding-2"

    # PostgreSQL + pgvector
    database_url: str = "postgresql+asyncpg://petro:petro@localhost:5432/petroglifos"
    database_url_sync: str = "postgresql+psycopg2://petro:petro@localhost:5432/petroglifos"

    # RAG
    rag_top_k: int = 5
    rag_min_similarity: float = 0.55
    confidence_threshold: float = 0.70
    rag_ingest_enabled: bool = False

    # Object storage
    storage_bucket: str = "petroglifos"
    storage_endpoint: str = "http://localhost:9000"
    storage_access_key: str = "minioadmin"
    storage_secret_key: str = "minioadmin"
    # Redis / Celery
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str | None = None
    celery_result_backend: str | None = None

    # Reconstruction API (petroglyph-service-reconstruction-api)
    # En Docker: http://reconstruction:8001  |  Local: http://localhost:8001
    reconstruction_api_base_url: str = "http://localhost:8001"
    reconstruction_visual_assisted_url: str = "http://localhost:8001/reconstructVisualAssisted"
    gan_api_url: str = "http://localhost:8001/reconstruct"
    gan_api_key: str = ""
    gan_mock_mode: bool = True

    # Telegram
    telegram_bot_token: str = ""

    # App
    env: str = "development"
    log_level: str = "INFO"
    max_record_sheet_minutes: int = 45
    # Grafo social
    edge_reliable_min_similarity: float = 0.70
    edge_min_evidence: int = 2
    # URL base de la API FastAPI — usada por el bot de Telegram
    # En Docker: http://api:8000  |  Local: http://localhost:8000
    api_base_url: str = "http://localhost:8000"

    @model_validator(mode="after")
    def _fill_redis_defaults(self) -> "Settings":
        """Permite usar REDIS_URL como fuente de verdad para Celery."""
        if not self.celery_broker_url:
            self.celery_broker_url = self.redis_url
        if not self.celery_result_backend:
            self.celery_result_backend = self.redis_url

        # Mantener alineadas las URLs del servicio de reconstrucción en local.
        # Si GAN_API_URL apunta a otro puerto/host, derivamos el base URL y el
        # endpoint visual asistido desde esa misma raíz.
        parsed_gan = urlsplit(self.gan_api_url)
        if parsed_gan.scheme and parsed_gan.netloc:
            derived_base = f"{parsed_gan.scheme}://{parsed_gan.netloc}"
            if self.reconstruction_api_base_url == "http://localhost:8001":
                self.reconstruction_api_base_url = derived_base
            if self.reconstruction_visual_assisted_url == "http://localhost:8001/reconstructVisualAssisted":
                self.reconstruction_visual_assisted_url = (
                    f"{self.reconstruction_api_base_url.rstrip('/')}/reconstructVisualAssisted"
                )
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
