-- ============================================================
-- 001_initial_schema.sql
-- Sistema de Petroglifos — esquema inicial completo
-- ============================================================

-- Extensión vectorial
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ── Sitios rupestres ─────────────────────────────────────────
CREATE TABLE rupestrian_sites (
    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name                TEXT NOT NULL,
    municipality        TEXT NOT NULL DEFAULT '',
    department          TEXT NOT NULL DEFAULT '',
    latitude            DOUBLE PRECISION,
    longitude           DOUBLE PRECISION,
    conservation_status TEXT DEFAULT 'Regular',
    dominant_taxonomy   TEXT DEFAULT 'Indeterminado',
    petroglyph_count    INTEGER DEFAULT 0,
    metadata            JSONB DEFAULT '{}',
    created_at          TIMESTAMPTZ DEFAULT now()
);

-- ── Petroglifos ───────────────────────────────────────────────
CREATE TABLE petroglyphs (
    id                       UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_id                  UUID REFERENCES rupestrian_sites(id),
    raw_image_path           TEXT DEFAULT '',
    preprocessed_image_path  TEXT DEFAULT '',
    reconstructed_image_path TEXT DEFAULT '',
    motif_description        TEXT DEFAULT '',
    detected_shapes          JSONB DEFAULT '[]',
    conservation_status      TEXT DEFAULT 'Regular',
    researcher_notes         TEXT DEFAULT '',
    created_at               TIMESTAMPTZ DEFAULT now(),
    updated_at               TIMESTAMPTZ DEFAULT now()
);

-- ── Corpus RAG: fragmentos documentales ──────────────────────
CREATE TABLE archaeological_chunks (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_document TEXT NOT NULL,
    chunk_text      TEXT NOT NULL,
    embedding       VECTOR(768),
    chunk_index     INTEGER DEFAULT 0,
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ix_arch_chunks_embedding
    ON archaeological_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- ── Embeddings de imágenes (comparador A3) ───────────────────
CREATE TABLE image_embeddings (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id   UUID REFERENCES petroglyphs(id),
    site_name       TEXT DEFAULT '',
    municipality    TEXT DEFAULT '',
    reference_name  TEXT DEFAULT '',
    taxonomy        TEXT DEFAULT 'Indeterminado',
    image_path      TEXT DEFAULT '',
    embedding       VECTOR(1280),
    metadata        JSONB DEFAULT '{}',
    created_at      TIMESTAMPTZ DEFAULT now()
);
CREATE INDEX ix_img_embeddings_embedding
    ON image_embeddings
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 50);

-- ── Clasificaciones generadas por A4 ─────────────────────────
CREATE TABLE llm_classifications (
    id                   UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id        UUID NOT NULL REFERENCES petroglyphs(id),
    taxonomy             TEXT NOT NULL,
    confidence           FLOAT DEFAULT 0.0,
    justification        TEXT DEFAULT '',
    retrieved_context    JSONB DEFAULT '[]',
    similarity_matches   JSONB DEFAULT '[]',
    requires_validation  BOOLEAN DEFAULT TRUE,
    low_context_quality  BOOLEAN DEFAULT FALSE,
    status               TEXT DEFAULT 'success',
    created_at           TIMESTAMPTZ DEFAULT now()
);

-- ── Logs de prompts enviados a Gemini ────────────────────────
CREATE TABLE prompt_logs (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id  UUID,
    prompt         TEXT DEFAULT '',
    response       TEXT DEFAULT '',
    tokens_input   INTEGER DEFAULT 0,
    tokens_output  INTEGER DEFAULT 0,
    latency_ms     INTEGER DEFAULT 0,
    status_code    TEXT DEFAULT 'ok',
    created_at     TIMESTAMPTZ DEFAULT now()
);

-- ── Fichas ICANH generadas ────────────────────────────────────
CREATE TABLE icanh_records (
    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    petroglyph_id           UUID NOT NULL REFERENCES petroglyphs(id),
    content_json            JSONB DEFAULT '{}',
    pdf_path                TEXT DEFAULT '',
    generation_date         TIMESTAMPTZ DEFAULT now(),
    generation_time_seconds FLOAT DEFAULT 0.0,
    expert_validation_status TEXT DEFAULT 'pending'
);

-- ── Grafo social: aristas de similitud entre sitios ──────────
CREATE TABLE site_graph_edges (
    id               UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    site_a_id        UUID NOT NULL REFERENCES rupestrian_sites(id),
    site_b_id        UUID NOT NULL REFERENCES rupestrian_sites(id),
    weight           FLOAT DEFAULT 0.0,
    shared_taxonomies JSONB DEFAULT '[]',
    evidence_count   INTEGER DEFAULT 1,
    created_at       TIMESTAMPTZ DEFAULT now(),
    updated_at       TIMESTAMPTZ DEFAULT now(),
    CONSTRAINT uq_site_graph_edge UNIQUE (site_a_id, site_b_id)
);

-- ── Función para actualizar updated_at ────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN NEW.updated_at = now(); RETURN NEW; END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_petroglyphs_updated_at
    BEFORE UPDATE ON petroglyphs
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();

CREATE TRIGGER trg_site_graph_edges_updated_at
    BEFORE UPDATE ON site_graph_edges
    FOR EACH ROW EXECUTE FUNCTION update_updated_at();