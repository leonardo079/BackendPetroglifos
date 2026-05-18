"""Búsqueda semántica top-k con similitud coseno sobre pgvector."""
from __future__ import annotations
import structlog
from adapters.outbound.embeddings.gemini_embedding_adapter import GeminiEmbeddingAdapter
from adapters.outbound.vector_store.pgvector_adapter import PgvectorAdapter
from config.settings import settings

log = structlog.get_logger(__name__)


class RAGRetriever:
    """
    Orquesta la búsqueda RAG completa:
    texto → embedding → pgvector → fragmentos relevantes.
    """

    def __init__(
        self,
        embedder: GeminiEmbeddingAdapter,
        vector_store: PgvectorAdapter,
    ) -> None:
        self._embedder = embedder
        self._vector_store = vector_store

    async def retrieve(
        self,
        query: str,
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[dict]:
        """
        Retorna los fragmentos más relevantes para la consulta.

        Args:
            query: Texto de la consulta (descripción del motivo, formas detectadas, etc.)
            k: Número de fragmentos a recuperar.
            min_similarity: Umbral mínimo de similitud coseno.

        Returns:
            Lista de dicts con keys: id, text, source, similarity, metadata
        """
        query_vec = await self._embedder.embed_query(query)
        chunks = await self._vector_store.similarity_search(
            query_vector=query_vec,
            k=k or settings.rag_top_k,
            min_similarity=min_similarity or settings.rag_min_similarity,
        )

        if not chunks:
            log.warning("rag_no_results", query_preview=query[:80])

        log.info("rag_retrieved", query_preview=query[:60], results=len(chunks))
        return chunks

    async def retrieve_for_motif(
        self,
        motif_description: str,
        detected_shapes: list[str],
        taxonomy_hint: str = "",
    ) -> list[dict]:
        """
        Construye una consulta enriquecida para buscar contexto
        arqueológico sobre un motivo detectado.
        """
        parts = [f"Motivo rupestre: {motif_description}"]
        if detected_shapes:
            parts.append(f"Formas: {', '.join(detected_shapes)}")
        if taxonomy_hint:
            parts.append(f"Posible categoría: {taxonomy_hint}")
        parts.append("arte rupestre petroglifo andino colombiano Boyacá Cundinamarca Muisca")

        query = ". ".join(parts)
        return await self.retrieve(query)