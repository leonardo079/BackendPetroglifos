"""Modelos ORM — todas las tablas del sistema."""
from __future__ import annotations
from datetime import datetime
from uuid import uuid4
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from pgvector.sqlalchemy import Vector
from infrastructure.database.session import Base


def _uuid() -> str:
    return str(uuid4())


# ─── Corpus RAG ──────────────────────────────────────────────────────────────

class ArchaeologicalChunk(Base):
    __tablename__ = "archaeological_chunks"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    source_document: Mapped[str] = mapped_column(sa.Text, nullable=False)
    chunk_text: Mapped[str] = mapped_column(sa.Text, nullable=False)
    embedding: Mapped[list[float]] = mapped_column(Vector(1280), nullable=True)
    chunk_index: Mapped[int] = mapped_column(sa.Integer, default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    __table_args__ = (
        sa.Index("ix_arch_chunks_embedding", "embedding",
                 postgresql_using="ivfflat",
                 postgresql_with={"lists": 100},
                 postgresql_ops={"embedding": "vector_cosine_ops"}),
        # Restricción de unicidad para evitar duplicados al reingestar el mismo documento.
        # Permite usar INSERT ... ON CONFLICT DO NOTHING en el adaptador.
        sa.UniqueConstraint("source_document", "chunk_index", name="uq_chunk_source_index"),
    )


# ─── Sitios rupestres ─────────────────────────────────────────────────────────

class RupestranSiteModel(Base):
    __tablename__ = "rupestrian_sites"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    name: Mapped[str] = mapped_column(sa.String(255), nullable=False, unique=True)
    municipality: Mapped[str] = mapped_column(sa.String(255), default="")
    department: Mapped[str] = mapped_column(sa.String(255), default="")
    latitude: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    longitude: Mapped[float | None] = mapped_column(sa.Float, nullable=True)
    conservation_status: Mapped[str] = mapped_column(sa.String(50), default="Regular")
    dominant_taxonomy: Mapped[str] = mapped_column(sa.String(100), default="Indeterminado")
    petroglyph_count: Mapped[int] = mapped_column(sa.Integer, default=0)
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    petroglyphs: Mapped[list["PetroglyphModel"]] = relationship(back_populates="site_rel")


# ─── Petroglifos ─────────────────────────────────────────────────────────────

class PetroglyphModel(Base):
    __tablename__ = "petroglyphs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_id: Mapped[str | None] = mapped_column(sa.ForeignKey("rupestrian_sites.id"), nullable=True)
    raw_image_path: Mapped[str] = mapped_column(sa.Text, default="")
    preprocessed_image_path: Mapped[str] = mapped_column(sa.Text, default="")
    reconstructed_image_path: Mapped[str] = mapped_column(sa.Text, default="")
    motif_description: Mapped[str] = mapped_column(sa.Text, default="")
    detected_shapes: Mapped[list] = mapped_column(JSONB, default=list)
    conservation_status: Mapped[str] = mapped_column(sa.String(50), default="Regular")
    researcher_notes: Mapped[str] = mapped_column(sa.Text, default="")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    site_rel: Mapped["RupestranSiteModel | None"] = relationship(back_populates="petroglyphs")
    classifications: Mapped[list["LLMClassification"]] = relationship(back_populates="petroglyph")
    icanh_records: Mapped[list["ICANHRecordModel"]] = relationship(back_populates="petroglyph")


# ─── Embeddings de imágenes (para comparador A3) ──────────────────────────────

class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    petroglyph_id: Mapped[str | None] = mapped_column(sa.ForeignKey("petroglyphs.id"), nullable=True)
    site_name: Mapped[str] = mapped_column(sa.String(255), default="")
    municipality: Mapped[str] = mapped_column(sa.String(255), default="")
    reference_name: Mapped[str] = mapped_column(sa.String(255), default="")
    taxonomy: Mapped[str] = mapped_column(sa.String(100), default="Indeterminado")
    image_path: Mapped[str] = mapped_column(sa.Text, default="")
    embedding: Mapped[list[float]] = mapped_column(Vector(1280), nullable=True)  # EfficientNet-B0
    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    __table_args__ = (
        sa.Index("ix_img_embeddings_embedding", "embedding",
                 postgresql_using="ivfflat",
                 postgresql_with={"lists": 50},
                 postgresql_ops={"embedding": "vector_cosine_ops"}),
    )


# ─── Clasificaciones LLM ──────────────────────────────────────────────────────

class LLMClassification(Base):
    __tablename__ = "llm_classifications"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    petroglyph_id: Mapped[str] = mapped_column(sa.ForeignKey("petroglyphs.id"), nullable=False)
    taxonomy: Mapped[str] = mapped_column(sa.String(100), nullable=False)
    confidence: Mapped[float] = mapped_column(sa.Float, default=0.0)
    justification: Mapped[str] = mapped_column(sa.Text, default="")
    retrieved_context: Mapped[list] = mapped_column(JSONB, default=list)
    similarity_matches: Mapped[list] = mapped_column(JSONB, default=list)
    requires_validation: Mapped[bool] = mapped_column(sa.Boolean, default=True)
    low_context_quality: Mapped[bool] = mapped_column(sa.Boolean, default=False)
    status: Mapped[str] = mapped_column(sa.String(50), default="success")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    petroglyph: Mapped["PetroglyphModel"] = relationship(back_populates="classifications")


# ─── Descripciones enriquecidas de petroglifos (LLM + embedding) ────────────

class PetroglyphDescriptionEmbedding(Base):
    __tablename__ = "petroglyph_description_embeddings"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    petroglyph_id: Mapped[str] = mapped_column(sa.ForeignKey("petroglyphs.id"), nullable=False)
    taxonomy: Mapped[str] = mapped_column(sa.String(100), default="Indeterminado")
    detailed_description: Mapped[str] = mapped_column(sa.Text, default="")
    probable_site: Mapped[str] = mapped_column(sa.String(255), default="")
    site_probability: Mapped[float] = mapped_column(sa.Float, default=0.0)
    key_figure_info: Mapped[list] = mapped_column(JSONB, default=list)
    embedding: Mapped[list[float]] = mapped_column(Vector(1280), nullable=True)
    rag_feedback: Mapped[dict] = mapped_column(JSONB, default=dict)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)

    __table_args__ = (
        sa.Index("ix_petroglyph_desc_embedding", "embedding",
                 postgresql_using="ivfflat",
                 postgresql_with={"lists": 100},
                 postgresql_ops={"embedding": "vector_cosine_ops"}),
    )


# ─── Logs de prompts ──────────────────────────────────────────────────────────

class PromptLog(Base):
    __tablename__ = "prompt_logs"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    petroglyph_id: Mapped[str | None] = mapped_column(sa.String, nullable=True)
    prompt: Mapped[str] = mapped_column(sa.Text, default="")
    response: Mapped[str] = mapped_column(sa.Text, default="")
    tokens_input: Mapped[int] = mapped_column(sa.Integer, default=0)
    tokens_output: Mapped[int] = mapped_column(sa.Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(sa.Integer, default=0)
    status_code: Mapped[str] = mapped_column(sa.String(50), default="ok")
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)


# ─── Fichas ICANH ─────────────────────────────────────────────────────────────

class ICANHRecordModel(Base):
    __tablename__ = "icanh_records"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    petroglyph_id: Mapped[str] = mapped_column(sa.ForeignKey("petroglyphs.id"), nullable=False)
    content_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    pdf_path: Mapped[str] = mapped_column(sa.Text, default="")
    generation_date: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    generation_time_seconds: Mapped[float] = mapped_column(sa.Float, default=0.0)
    expert_validation_status: Mapped[str] = mapped_column(sa.String(50), default="pending")

    petroglyph: Mapped["PetroglyphModel"] = relationship(back_populates="icanh_records")


# ─── Aristas del grafo social ─────────────────────────────────────────────────

class SiteGraphEdge(Base):
    """Aristas del grafo de similitud iconográfica entre sitios."""
    __tablename__ = "site_graph_edges"

    id: Mapped[str] = mapped_column(UUID(as_uuid=False), primary_key=True, default=_uuid)
    site_a_id: Mapped[str] = mapped_column(sa.ForeignKey("rupestrian_sites.id"), nullable=False)
    site_b_id: Mapped[str] = mapped_column(sa.ForeignKey("rupestrian_sites.id"), nullable=False)
    weight: Mapped[float] = mapped_column(sa.Float, default=0.0)  # similitud coseno promedio
    shared_taxonomies: Mapped[list] = mapped_column(JSONB, default=list)
    evidence_count: Mapped[int] = mapped_column(sa.Integer, default=1)
    created_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(sa.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    __table_args__ = (
        sa.UniqueConstraint("site_a_id", "site_b_id", name="uq_site_graph_edge"),
    )
