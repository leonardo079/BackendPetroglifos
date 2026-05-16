-- Extensión vectorial
CREATE EXTENSION IF NOT EXISTS vector;

-- Fragmentos documentales del corpus arqueológico
CREATE TABLE archaeological_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document TEXT NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       VECTOR(768),
    metadata        JSONB,
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ON archaeological_chunks USING ivfflat (embedding vector_cosine_ops);

-- Clasificaciones generadas por A4
CREATE TABLE llm_classifications (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id        UUID NOT NULL,
    taxonomy             TEXT NOT NULL,
    confidence           FLOAT,
    justification        TEXT,
    retrieved_context    JSONB,
    requires_validation  BOOLEAN DEFAULT TRUE,
    low_context_quality  BOOLEAN DEFAULT FALSE,
    status               TEXT DEFAULT 'success',
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- Registro de prompts enviados a Gemini
CREATE TABLE prompt_logs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id  UUID,
    prompt         TEXT,
    response       TEXT,
    tokens_input   INTEGER,
    tokens_output  INTEGER,
    latency_ms     INTEGER,
    status_code    TEXT,
    created_at     TIMESTAMPTZ DEFAULT now()
);
