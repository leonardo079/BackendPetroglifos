"""Adaptador para PostgreSQL + pgvector."""
from core.ports.outbound.vector_store_port import VectorStorePort

class PgvectorAdapter(VectorStorePort):
    async def similarity_search(self, query_vector: list[float], k: int = 5) -> list[dict]:
        raise NotImplementedError
    async def upsert(self, documents: list[dict]) -> None:
        raise NotImplementedError
