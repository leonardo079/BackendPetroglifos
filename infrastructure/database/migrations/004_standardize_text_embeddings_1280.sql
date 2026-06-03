-- ============================================================
-- 004_standardize_text_embeddings_1280.sql
-- Ajusta las tablas de texto para usar 1280 dimensiones
-- ============================================================

DROP INDEX IF EXISTS ix_arch_chunks_embedding;
DROP INDEX IF EXISTS ix_petroglyph_desc_embedding;

ALTER TABLE archaeological_chunks
    ALTER COLUMN embedding TYPE VECTOR(1280)
    USING embedding::vector(1280);

ALTER TABLE petroglyph_description_embeddings
    ALTER COLUMN embedding TYPE VECTOR(1280)
    USING embedding::vector(1280);

CREATE INDEX IF NOT EXISTS ix_arch_chunks_embedding
    ON archaeological_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS ix_petroglyph_desc_embedding
    ON petroglyph_description_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);
