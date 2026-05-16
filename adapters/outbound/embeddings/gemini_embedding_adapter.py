"""Adaptador para Gemini text-embedding-004 (768 dimensiones)."""
from core.ports.outbound.embedding_port import EmbeddingPort

class GeminiEmbeddingAdapter(EmbeddingPort):
    async def embed(self, text: str) -> list[float]:
        raise NotImplementedError
