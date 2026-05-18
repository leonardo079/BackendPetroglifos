"""Pipeline de ingestión: PDF/OCR → chunks → embeddings → pgvector."""
from __future__ import annotations
import asyncio
import os
from pathlib import Path
import structlog
from pypdf import PdfReader
import pytesseract
from PIL import Image

from rag.chunking.semantic_chunker import chunk_text
from adapters.outbound.embeddings.gemini_embedding_adapter import GeminiEmbeddingAdapter
from adapters.outbound.vector_store.pgvector_adapter import PgvectorAdapter

log = structlog.get_logger(__name__)

# Pausa entre llamadas a la API de embeddings (rate limit: 15 req/min en free tier)
_EMBED_DELAY = 4.5  # segundos


class DocumentIngester:
    """
    Ingesta documentos (PDF, TXT, imagen) al vector store.

    Uso:
        ingester = DocumentIngester(embedding_adapter, vector_adapter)
        await ingester.ingest_file("/path/to/doc.pdf")
    """

    def __init__(
        self,
        embedding_adapter: GeminiEmbeddingAdapter,
        vector_adapter: PgvectorAdapter,
        chunk_size: int = 512,
        overlap: int = 50,
    ) -> None:
        self._embedder = embedding_adapter
        self._vector = vector_adapter
        self._chunk_size = chunk_size
        self._overlap = overlap

    # ── Extracción de texto ───────────────────────────────────────────────────

    def _extract_pdf(self, path: str) -> str:
        reader = PdfReader(path)
        pages: list[str] = []
        for page in reader.pages:
            text = page.extract_text() or ""
            if len(text.strip()) < 50:
                # Página probablemente escaneada — intentar OCR
                log.warning("pdf_page_sparse_text", path=path)
            pages.append(text)
        return "\n\n".join(pages)

    def _extract_image_ocr(self, path: str) -> str:
        img = Image.open(path)
        return pytesseract.image_to_string(img, lang="spa+eng")

    def _extract_txt(self, path: str) -> str:
        with open(path, encoding="utf-8", errors="replace") as f:
            return f.read()

    def _extract(self, path: str) -> str:
        ext = Path(path).suffix.lower()
        if ext == ".pdf":
            return self._extract_pdf(path)
        elif ext in {".png", ".jpg", ".jpeg", ".tiff", ".bmp"}:
            return self._extract_image_ocr(path)
        else:
            return self._extract_txt(path)

    # ── Pipeline principal ────────────────────────────────────────────────────

    async def ingest_file(self, path: str, source_name: str | None = None) -> int:
        """Ingesta un archivo. Retorna el número de chunks insertados."""
        source = source_name or Path(path).name
        log.info("ingesting_file", path=path, source=source)

        text = self._extract(path)
        if not text.strip():
            log.warning("empty_document", path=path)
            return 0

        chunks = chunk_text(text, source=source, chunk_size=self._chunk_size, overlap=self._overlap)
        enriched: list[dict] = []

        for i, chunk in enumerate(chunks):
            log.debug("embedding_chunk", chunk_index=i, total=len(chunks))
            embedding = await self._embedder.embed(chunk["text"])
            enriched.append({**chunk, "embedding": embedding})
            # Respetar rate limit del tier gratuito de Gemini
            if i < len(chunks) - 1:
                await asyncio.sleep(_EMBED_DELAY)

        await self._vector.upsert(enriched)
        log.info("ingest_complete", source=source, chunks=len(enriched))
        return len(enriched)

    async def ingest_directory(self, dir_path: str, extensions: tuple[str, ...] = (".pdf", ".txt")) -> dict:
        """Ingesta todos los documentos de un directorio."""
        results: dict[str, int] = {}
        files = [
            str(p) for p in Path(dir_path).rglob("*")
            if p.suffix.lower() in extensions
        ]
        log.info("ingesting_directory", dir=dir_path, files=len(files))
        for file_path in files:
            try:
                count = await self.ingest_file(file_path)
                results[file_path] = count
            except Exception as e:
                log.error("ingest_file_error", path=file_path, error=str(e))
                results[file_path] = -1
        return results