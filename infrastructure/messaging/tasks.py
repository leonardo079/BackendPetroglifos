"""
Tareas Celery para procesamiento asíncrono del pipeline de petroglifos.

Uso desde el API:
    from infrastructure.messaging.tasks import classify_petroglyph_task
    result = classify_petroglyph_task.delay(task_id, image_path, site_metadata)
    # Polling: result.get(timeout=2700)
"""
from __future__ import annotations
import asyncio
import structlog
from infrastructure.messaging.celery_app import celery_app

log = structlog.get_logger(__name__)


def _run_async(coro):
    """Ejecuta una corrutina desde un contexto síncrono (necesario en Celery)."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            raise RuntimeError
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


@celery_app.task(
    bind=True,
    name="petroglifos.classify",
    queue="petroglifos",
    max_retries=3,
    default_retry_delay=30,
    acks_late=True,
)
def classify_petroglyph_task(
    self,
    task_id: str,
    image_path: str,
    site_metadata: dict,
) -> dict:
    """
    Tarea principal: ejecuta el pipeline completo de clasificación.

    Args:
        task_id: ID único de la tarea (UUID).
        image_path: Ruta local a la imagen del petroglifo.
        site_metadata: Diccionario con nombre del sitio, municipio, GPS, etc.

    Returns:
        Resultado de la clasificación (taxonomy, confidence, pdf_url…).
    """
    log.info("celery_task_start", task_id=task_id, image=image_path)
    try:
        result = _run_async(_run_pipeline(task_id, image_path, site_metadata))
        log.info("celery_task_done", task_id=task_id, status=result.get("status"))
        return result
    except Exception as exc:
        log.error("celery_task_error", task_id=task_id, error=str(exc))
        raise self.retry(exc=exc)


async def _run_pipeline(task_id: str, image_path: str, site_metadata: dict) -> dict:
    """Importa y ejecuta el orquestador dentro de un contexto async."""
    from infrastructure.database.session import AsyncSessionLocal
    from orchestrator.PetroglyphOrchestrator import create_orchestrator

    async with AsyncSessionLocal() as session:
        orchestrator = await create_orchestrator(session)
        return await orchestrator.run(task_id, image_path, site_metadata)


@celery_app.task(
    bind=True,
    name="petroglifos.ingest_document",
    queue="ingestion",
    max_retries=2,
    default_retry_delay=60,
)
def ingest_document_task(self, file_path: str, source_name: str | None = None) -> dict:
    """
    Tarea de ingestión: procesa un documento y lo carga al vector store.

    Args:
        file_path: Ruta al archivo (PDF, TXT, imagen).
        source_name: Nombre del documento fuente (opcional).

    Returns:
        Número de chunks insertados y estado.
    """
    log.info("ingest_task_start", file=file_path)
    try:
        chunks = _run_async(_run_ingestion(file_path, source_name))
        log.info("ingest_task_done", file=file_path, chunks=chunks)
        return {"status": "success", "chunks_inserted": chunks, "file": file_path}
    except Exception as exc:
        log.error("ingest_task_error", file=file_path, error=str(exc))
        raise self.retry(exc=exc)


async def _run_ingestion(file_path: str, source_name: str | None) -> int:
    from infrastructure.database.session import AsyncSessionLocal
    from adapters.outbound.embeddings.gemini_embedding_adapter import GeminiEmbeddingAdapter
    from adapters.outbound.vector_store.pgvector_adapter import PgvectorAdapter
    from rag.ingestion.document_ingester import DocumentIngester

    async with AsyncSessionLocal() as session:
        embedder = GeminiEmbeddingAdapter()
        vector_store = PgvectorAdapter(session)
        ingester = DocumentIngester(embedder, vector_store)
        return await ingester.ingest_file(file_path, source_name)