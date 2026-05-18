"""Adaptador para Gemini text-embedding-004 (768 dimensiones)."""
from __future__ import annotations
import time
import structlog
import google.generativeai as genai
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from core.ports.outbound.embedding_port import EmbeddingPort
from config.settings import settings

log = structlog.get_logger(__name__)


class GeminiEmbeddingAdapter(EmbeddingPort):
    """Genera embeddings de 768 dims con text-embedding-004."""

    def __init__(self) -> None:
        genai.configure(api_key=settings.gemini_api_key)
        self._model = settings.embedding_model

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed(self, text: str) -> list[float]:
        """Genera un embedding para un texto dado."""
        t0 = time.monotonic()
        result = genai.embed_content(
            model=f"models/{self._model}",
            content=text,
            task_type="retrieval_document",
        )
        elapsed = (time.monotonic() - t0) * 1000
        log.debug("embedding_generated", model=self._model, chars=len(text), latency_ms=round(elapsed))
        return result["embedding"]

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_query(self, text: str) -> list[float]:
        """Genera un embedding optimizado para búsqueda (query)."""
        result = genai.embed_content(
            model=f"models/{self._model}",
            content=text,
            task_type="retrieval_query",
        )
        return result["embedding"]

    async def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Procesa una lista de textos en lotes."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            log.info("embedding_batch", batch_num=i // batch_size + 1, size=len(batch))
            for text in batch:
                emb = await self.embed(text)
                embeddings.append(emb)
        return embeddings