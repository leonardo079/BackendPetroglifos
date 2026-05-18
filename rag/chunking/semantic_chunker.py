"""Chunking semántico: 512 tokens, overlap 50."""
from __future__ import annotations
import re
import structlog

log = structlog.get_logger(__name__)

# Aprox chars por token en español
CHARS_PER_TOKEN = 4


def _count_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[dict]:
    """
    Divide texto en fragmentos semánticos con overlap.
    Respeta párrafos siempre que sea posible.
    """
    # Limpiar texto
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    paragraphs = text.split("\n\n")

    chunks: list[dict] = []
    current_tokens: list[str] = []
    current_size = 0
    chunk_index = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        para_tokens = _count_tokens(para)

        if current_size + para_tokens > chunk_size and current_tokens:
            chunk_text_str = "\n\n".join(current_tokens)
            chunks.append({
                "text": chunk_text_str,
                "source": source,
                "chunk_index": chunk_index,
                "metadata": {"token_count": current_size},
            })
            chunk_index += 1
            # Overlap: mantener último fragmento si cabe
            if overlap > 0 and current_tokens:
                last = current_tokens[-1]
                last_tokens = _count_tokens(last)
                if last_tokens <= overlap:
                    current_tokens = [last]
                    current_size = last_tokens
                else:
                    current_tokens = []
                    current_size = 0
            else:
                current_tokens = []
                current_size = 0

        current_tokens.append(para)
        current_size += para_tokens

    # Último chunk
    if current_tokens:
        chunks.append({
            "text": "\n\n".join(current_tokens),
            "source": source,
            "chunk_index": chunk_index,
            "metadata": {"token_count": current_size},
        })

    log.info("chunking_complete", source=source, chunks=len(chunks))
    return chunks