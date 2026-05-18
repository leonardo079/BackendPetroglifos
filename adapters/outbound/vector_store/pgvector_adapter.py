"""Adaptador para PostgreSQL + pgvector — búsqueda semántica."""
from __future__ import annotations
import structlog
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from core.ports.outbound.vector_store_port import VectorStorePort
from infrastructure.database.models.models import ArchaeologicalChunk, ImageEmbedding
from config.settings import settings

log = structlog.get_logger(__name__)


class PgvectorAdapter(VectorStorePort):
    """Búsqueda de similitud coseno sobre archaeological_chunks."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def similarity_search(
        self,
        query_vector: list[float],
        k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[dict]:
        """Retorna los k fragmentos más similares al vector de consulta."""
        k = k or settings.rag_top_k
        min_sim = min_similarity if min_similarity is not None else settings.rag_min_similarity

        sql = text("""
            SELECT
                id,
                source_document,
                chunk_text,
                metadata,
                1 - (embedding <=> :query_vec::vector) AS similarity
            FROM archaeological_chunks
            WHERE 1 - (embedding <=> :query_vec::vector) >= :min_sim
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :k
        """)
        result = await self._session.execute(
            sql,
            {
                "query_vec": str(query_vector),
                "min_sim": min_sim,
                "k": k,
            },
        )
        rows = result.fetchall()
        log.debug("pgvector_search", k=k, results=len(rows), min_similarity=min_sim)
        return [
            {
                "id": str(row.id),
                "text": row.chunk_text,
                "source": row.source_document,
                "similarity": float(row.similarity),
                "metadata": row.metadata or {},
            }
            for row in rows
        ]

    async def upsert(self, documents: list[dict]) -> None:
        """
        Inserta fragmentos con sus embeddings en archaeological_chunks.

        Usa INSERT ... ON CONFLICT DO NOTHING para que reingestar el mismo
        documento sea idempotente: los chunks ya existentes se omiten silenciosamente.
        Requiere la restricción uq_chunk_source_index en el modelo.
        """
        from sqlalchemy.dialects.postgresql import insert as pg_insert

        inserted = 0
        skipped = 0
        for doc in documents:
            stmt = (
                pg_insert(ArchaeologicalChunk)
                .values(
                    source_document=doc["source"],
                    chunk_text=doc["text"],
                    embedding=doc["embedding"],
                    chunk_index=doc.get("chunk_index", 0),
                    metadata_=doc.get("metadata", {}),
                )
                .on_conflict_do_nothing(constraint="uq_chunk_source_index")
            )
            result = await self._session.execute(stmt)
            if result.rowcount:
                inserted += 1
            else:
                skipped += 1
        await self._session.commit()
        log.info("pgvector_upsert", inserted=inserted, skipped=skipped, total=len(documents))


class ImageVectorAdapter:
    """Búsqueda de similitud sobre image_embeddings (EfficientNet-B0, 1280 dims)."""

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def similarity_search(
        self,
        query_vector: list[float],
        k: int = 5,
        min_similarity: float = 0.60,
    ) -> list[dict]:
        sql = text("""
            SELECT
                id,
                petroglyph_id,
                site_name,
                municipality,
                reference_name,
                taxonomy,
                image_path,
                1 - (embedding <=> :query_vec::vector) AS similarity
            FROM image_embeddings
            WHERE 1 - (embedding <=> :query_vec::vector) >= :min_sim
            ORDER BY embedding <=> :query_vec::vector
            LIMIT :k
        """)
        result = await self._session.execute(
            sql,
            {"query_vec": str(query_vector), "min_sim": min_similarity, "k": k},
        )
        rows = result.fetchall()
        return [
            {
                "id": str(row.id),
                "petroglyph_id": str(row.petroglyph_id) if row.petroglyph_id else None,
                "site_name": row.site_name,
                "municipality": row.municipality,
                "reference_name": row.reference_name,
                "taxonomy": row.taxonomy,
                "image_path": row.image_path,
                "similarity_score": float(row.similarity),
            }
            for row in rows
        ]

    async def upsert(self, records: list[dict]) -> None:
        for rec in records:
            img_emb = ImageEmbedding(
                petroglyph_id=rec.get("petroglyph_id"),
                site_name=rec.get("site_name", ""),
                municipality=rec.get("municipality", ""),
                reference_name=rec.get("reference_name", ""),
                taxonomy=rec.get("taxonomy", "Indeterminado"),
                image_path=rec.get("image_path", ""),
                embedding=rec["embedding"],
                metadata_=rec.get("metadata", {}),
            )
            self._session.add(img_emb)
        await self._session.commit()
        log.info("image_embeddings_upsert", count=len(records))