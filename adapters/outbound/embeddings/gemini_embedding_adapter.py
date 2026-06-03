"""Adaptador para embeddings con Gemini y fallback local."""
from __future__ import annotations

import asyncio
import hashlib
import math
import re
import time

import structlog
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from config.settings import settings
from core.ports.outbound.embedding_port import EmbeddingPort

log = structlog.get_logger(__name__)


class GeminiEmbeddingAdapter(EmbeddingPort):
    """Genera embeddings y cae a un vector local si Gemini falla."""

    _DIMENSIONS = 1280

    def __init__(self) -> None:
        self._model = self._normalize_model_name(settings.embedding_model)
        self._embeddings = GoogleGenerativeAIEmbeddings(
            model=self._model,
            google_api_key=settings.gemini_api_key,
        )

    @staticmethod
    def _normalize_model_name(model_name: str) -> str:
        normalized = model_name.strip().removeprefix("models/")
        aliases = {
            "text-embedding-004": "gemini-embedding-001",
            "embedding-001": "gemini-embedding-001",
        }
        return aliases.get(normalized, normalized)

    @classmethod
    def _local_embedding(cls, text: str) -> list[float]:
        """Embedding hash-based determinista para operar sin Gemini."""
        vector = [0.0] * cls._DIMENSIONS
        tokens = re.findall(r"\w+", text.lower())
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "big") % cls._DIMENSIONS
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[bucket] += sign

        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0.0:
            return vector
        return [value / norm for value in vector]

    @classmethod
    def _coerce_dimensions(cls, embedding: list[float]) -> list[float]:
        """Normaliza el tamaño del vector al esperado por la BD."""
        if len(embedding) == cls._DIMENSIONS:
            return embedding
        if len(embedding) > cls._DIMENSIONS:
            return embedding[: cls._DIMENSIONS]
        return embedding + [0.0] * (cls._DIMENSIONS - len(embedding))

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed(self, text: str) -> list[float]:
        """Genera un embedding para un texto dado."""
        t0 = time.monotonic()
        try:
            result = await asyncio.to_thread(self._embeddings.embed_documents, [text])
            embedding = self._coerce_dimensions(result[0] if result else [])
            source = "gemini"
        except Exception as exc:
            log.warning("embedding_fallback_local", error=str(exc), model=self._model)
            embedding = self._coerce_dimensions(self._local_embedding(text))
            source = "local_hash"

        elapsed = (time.monotonic() - t0) * 1000
        log.debug(
            "embedding_generated",
            model=self._model,
            chars=len(text),
            latency_ms=round(elapsed),
            source=source,
        )
        return embedding

    @retry(
        retry=retry_if_exception_type(Exception),
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        reraise=True,
    )
    async def embed_query(self, text: str) -> list[float]:
        """Genera un embedding optimizado para búsqueda (query)."""
        try:
            embedding = await asyncio.to_thread(self._embeddings.embed_query, text)
            return self._coerce_dimensions(embedding)
        except Exception as exc:
            log.warning("embedding_query_fallback_local", error=str(exc), model=self._model)
            return self._coerce_dimensions(self._local_embedding(text))

    async def embed_batch(self, texts: list[str], batch_size: int = 20) -> list[list[float]]:
        """Procesa una lista de textos en lotes."""
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            log.info("embedding_batch", batch_num=i // batch_size + 1, size=len(batch))
            try:
                batch_embeddings = await asyncio.to_thread(self._embeddings.embed_documents, batch)
            except Exception as exc:
                log.warning("embedding_batch_fallback_local", error=str(exc), batch_size=len(batch))
                batch_embeddings = [self._local_embedding(text) for text in batch]
            embeddings.extend([self._coerce_dimensions(embedding) for embedding in batch_embeddings])
        return embeddings
