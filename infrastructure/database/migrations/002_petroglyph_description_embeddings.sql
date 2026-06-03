-- ============================================================
-- 002_petroglyph_description_embeddings.sql
-- Tabla separada para descripcion LLM del petroglifo + embedding
-- ============================================================

CREATE TABLE IF NOT EXISTS petroglyph_description_embeddings (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id        UUID NOT NULL REFERENCES petroglyphs(id),
    taxonomy             TEXT DEFAULT 'Indeterminado',
    detailed_description TEXT DEFAULT '',
    probable_site        TEXT DEFAULT '',
    site_probability     FLOAT DEFAULT 0.0,
    key_figure_info      JSONB DEFAULT '[]',
    embedding            VECTOR(1280),
    rag_feedback         JSONB DEFAULT '{}',
    created_at           TIMESTAMPTZ DEFAULT now()
);

CREATE INDEX IF NOT EXISTS ix_petroglyph_desc_embedding
    ON petroglyph_description_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
