"""
API FastAPI del Sistema de Petroglifos.

Endpoints:
    POST /classify                      — Clasifica un petroglifo (async vía Celery)
    POST /classify/sync                 — Clasifica de forma síncrona (para pruebas)
    GET  /tasks/{task_id}               — Consulta estado de una tarea Celery
    POST /ingest                        — Ingesta un documento al corpus RAG
    GET  /sites                         — Lista sitios rupestres registrados
    GET  /sites/{site_id}               — Detalle de un sitio (con conexiones del grafo)
    GET  /graph                         — Red social de similitud iconográfica (JSON)
    GET  /graph/export                  — HTML interactivo del grafo (PyVis)
    GET  /graph/export/image            — Imagen estática del grafo (PNG)
    GET  /graph/pagerank                — Ranking de centralidad PageRank por sitio
    GET  /graph/communities             — Comunidades iconográficas (Louvain)
    GET  /graph/betweenness             — Centralidad de intermediación (sitios puente)
    GET  /graph/sites/{site_id}/similar — Sitios más similares a uno dado
    GET  /health                        — Health check
"""
from __future__ import annotations
import uuid
from pathlib import Path
from typing import Annotated

import structlog
from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import settings
from core.domain.site_normalization import normalize_site_metadata
from infrastructure.database.session import get_session
from infrastructure.observability.logging_config import configure_logging

configure_logging()
log = structlog.get_logger(__name__)

app = FastAPI(
    title="Petroglifos LLM API",
    version="0.2.0",
    description="Sistema multiagente para clasificación taxonómica de petroglifos andinos colombianos — UPTC 2026",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

GRAPH_OUTPUT = Path("storage/graphs/red_rupestre.html")


# ── Schemas ────────────────────────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    image_path: str
    site: str = "Sin nombre"
    municipality: str = ""
    department: str = ""
    gps_coordinates: dict = {}
    conservation_status: str = "Regular"
    researcher_notes: str = ""
    petroglyph_id: str | None = None


class ClassifyResponse(BaseModel):
    task_id: str
    status: str
    classification: dict = {}
    icanh_pdf_url: str = ""
    total_time_ms: int = 0
    message: str = ""


class TaskStatusResponse(BaseModel):
    task_id: str
    celery_state: str
    result: dict | None = None


class IngestResponse(BaseModel):
    status: str
    chunks_inserted: int
    source: str


class SiteResponse(BaseModel):
    id: str
    name: str
    municipality: str
    department: str
    dominant_taxonomy: str
    petroglyph_count: int
    conservation_status: str


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Sistema"])
async def health_check(session: AsyncSession = Depends(get_session)) -> dict:
    """Verifica que la API y la base de datos estén operativas."""
    try:
        await session.execute(text("SELECT 1"))
        db_status = "ok"
    except Exception as e:
        db_status = f"error: {e}"

    return {
        "status": "ok" if db_status == "ok" else "degraded",
        "version": "0.2.0",
        "environment": settings.env,
        "database": db_status,
    }


# ── Clasificación asíncrona (Celery) ───────────────────────────────────────────

@app.post("/classify", response_model=ClassifyResponse, tags=["Clasificación"])
async def classify_async(payload: ClassifyRequest) -> ClassifyResponse:
    """
    Encola una tarea de clasificación en Celery.
    Retorna el task_id para consultar el estado con GET /tasks/{task_id}.
    """
    from infrastructure.messaging.tasks import classify_petroglyph_task

    task_id = payload.petroglyph_id or str(uuid.uuid4())
    site, municipality, department = normalize_site_metadata(
        payload.site,
        payload.municipality,
        payload.department,
    )
    site_metadata = {
        "site": site,
        "municipality": municipality,
        "department": department,
        "gps_coordinates": payload.gps_coordinates,
        "conservation_status": payload.conservation_status,
        "researcher_notes": payload.researcher_notes,
    }

    task = classify_petroglyph_task.apply_async(
        args=[task_id, payload.image_path, site_metadata],
        task_id=task_id,
    )
    log.info("api_classify_enqueued", task_id=task_id, site=payload.site)
    return ClassifyResponse(
        task_id=task_id,
        status="queued",
        message="Tarea encolada. Use GET /tasks/{task_id} para consultar el estado.",
    )


@app.post("/classify/upload", response_model=ClassifyResponse, tags=["Clasificación"])
async def classify_with_upload(
    site: str = Form("Sin nombre"),
    municipality: str = Form(""),
    department: str = Form(""),
    conservation_status: str = Form("Regular"),
    researcher_notes: str = Form(""),
    file: UploadFile = File(...),
) -> ClassifyResponse:
    """
    Acepta una imagen subida directamente y la encola para clasificación.
    Equivalente al flujo del bot de Telegram para uso web.
    """
    from infrastructure.messaging.tasks import classify_petroglyph_task

    task_id = str(uuid.uuid4())
    suffix = Path(file.filename or "image.jpg").suffix or ".jpg"
    dest = UPLOAD_DIR / f"{task_id}{suffix}"
    dest.write_bytes(await file.read())

    normalized_site, normalized_municipality, normalized_department = normalize_site_metadata(
        site,
        municipality,
        department,
    )
    site_metadata = {
        "site": normalized_site,
        "municipality": normalized_municipality,
        "department": normalized_department,
        "conservation_status": conservation_status,
        "researcher_notes": researcher_notes,
    }
    classify_petroglyph_task.apply_async(
        args=[task_id, str(dest), site_metadata],
        task_id=task_id,
    )
    log.info(
        "api_classify_upload_enqueued",
        task_id=task_id,
        filename=file.filename,
    )
    return ClassifyResponse(
        task_id=task_id,
        status="queued",
        message="Imagen recibida y tarea encolada.",
    )


@app.post("/classify/sync", response_model=ClassifyResponse, tags=["Clasificación"])
async def classify_sync(
    payload: ClassifyRequest,
    session: AsyncSession = Depends(get_session),
) -> ClassifyResponse:
    """
    Clasifica de forma síncrona (espera el resultado antes de responder).
    Solo para uso en desarrollo o pruebas unitarias.
    """
    from orchestrator.PetroglyphOrchestrator import create_orchestrator

    task_id = payload.petroglyph_id or str(uuid.uuid4())
    site, municipality, department = normalize_site_metadata(
        payload.site,
        payload.municipality,
        payload.department,
    )
    site_metadata = {
        "site": site,
        "municipality": municipality,
        "department": department,
        "gps_coordinates": payload.gps_coordinates,
        "conservation_status": payload.conservation_status,
        "researcher_notes": payload.researcher_notes,
    }

    orchestrator = await create_orchestrator(session)
    result = await orchestrator.run(task_id, payload.image_path, site_metadata)

    return ClassifyResponse(
        task_id=task_id,
        status=result.get("status", "error"),
        classification=result.get("classification", {}),
        icanh_pdf_url=result.get("icanh_pdf_url", ""),
        total_time_ms=result.get("total_time_ms", 0),
    )


# ── Estado de tareas Celery ───────────────────────────────────────────────────

@app.get("/tasks/{task_id}", response_model=TaskStatusResponse, tags=["Clasificación"])
async def get_task_status(task_id: str) -> TaskStatusResponse:
    """Consulta el estado y resultado de una tarea Celery."""
    from infrastructure.messaging.celery_app import celery_app

    result = celery_app.AsyncResult(task_id)
    payload = None
    if result.ready() and result.successful():
        payload = result.result
    return TaskStatusResponse(
        task_id=task_id,
        celery_state=result.state,
        result=payload,
    )


# ── Ingestión de corpus ───────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Corpus RAG"])
async def ingest_document(
    source_name: str = Form(...),
    file: UploadFile = File(...),
) -> IngestResponse:
    """
    Ingesta un documento PDF o TXT al corpus RAG (pgvector).
    La ingestión se realiza en background vía Celery.
    """
    from infrastructure.messaging.tasks import ingest_document_task

    dest = UPLOAD_DIR / f"corpus_{uuid.uuid4().hex}_{file.filename}"
    dest.write_bytes(await file.read())

    ingest_document_task.apply_async(args=[str(dest), source_name])
    log.info("api_ingest_enqueued", source=source_name, file=file.filename)
    return IngestResponse(status="queued", chunks_inserted=0, source=source_name)


# ── Sitios rupestres ──────────────────────────────────────────────────────────

@app.get("/sites", response_model=list[SiteResponse], tags=["Sitios"])
async def list_sites(
    session: AsyncSession = Depends(get_session),
    department: str | None = None,
    municipality: str | None = None,
) -> list[SiteResponse]:
    """Lista todos los sitios rupestres registrados. Soporta filtro por departamento y municipio."""
    from infrastructure.database.models.models import RupestranSiteModel

    stmt = select(RupestranSiteModel)
    if department:
        stmt = stmt.where(RupestranSiteModel.department.ilike(f"%{department}%"))
    if municipality:
        stmt = stmt.where(RupestranSiteModel.municipality.ilike(f"%{municipality}%"))

    result = await session.execute(stmt)
    sites = result.scalars().all()
    return [
        SiteResponse(
            id=s.id,
            name=s.name,
            municipality=s.municipality,
            department=s.department,
            dominant_taxonomy=s.dominant_taxonomy,
            petroglyph_count=s.petroglyph_count,
            conservation_status=s.conservation_status,
        )
        for s in sites
    ]


@app.get("/sites/{site_id}", tags=["Sitios"])
async def get_site(
    site_id: str,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """Retorna detalle de un sitio incluyendo sus petroglifos y similitudes iconográficas."""
    from infrastructure.database.models.models import RupestranSiteModel, SiteGraphEdge

    result = await session.execute(
        select(RupestranSiteModel).where(RupestranSiteModel.id == site_id)
    )
    site = result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail=f"Sitio {site_id} no encontrado.")

    # Aristas del grafo
    edges_result = await session.execute(
        select(SiteGraphEdge).where(
            (SiteGraphEdge.site_a_id == site_id) | (SiteGraphEdge.site_b_id == site_id)
        )
    )
    edges = edges_result.scalars().all()
    connections = [
        {
            "connected_site_id": e.site_b_id if e.site_a_id == site_id else e.site_a_id,
            "weight": e.weight,
            "evidence_count": e.evidence_count,
            "shared_taxonomies": e.shared_taxonomies,
        }
        for e in edges
    ]

    return {
        "id": site.id,
        "name": site.name,
        "municipality": site.municipality,
        "department": site.department,
        "latitude": site.latitude,
        "longitude": site.longitude,
        "dominant_taxonomy": site.dominant_taxonomy,
        "petroglyph_count": site.petroglyph_count,
        "conservation_status": site.conservation_status,
        "iconographic_connections": connections,
    }


# ── Grafo social ───────────────────────────────────────────────────────────────

async def _build_graph_from_db(session: AsyncSession):
    """
    Reconstruye el PetroglyphSocialGraph completo desde la BD.

    Usa el nombre del sitio como ID de nodo (no el UUID) para que los edges
    reconstruidos desde site_graph_edges conecten correctamente con los nodos.
    Preserva también weight, evidence_count e is_provisional.
    """
    from infrastructure.database.models.models import RupestranSiteModel, SiteGraphEdge
    from orchestrator.graph.petroglyph_graph import PetroglyphSocialGraph

    sites = list((await session.execute(select(RupestranSiteModel))).scalars().all())
    edges = list((await session.execute(select(SiteGraphEdge))).scalars().all())

    # Índice UUID → nombre de sitio para reconstruir edges sin mismatch
    id_to_name: dict = {s.id: s.name for s in sites}

    graph = PetroglyphSocialGraph()
    for site in sites:
        graph.add_site(
            site.name,
            municipality=site.municipality,
            department=site.department,
            dominant_taxonomy=site.dominant_taxonomy,
            petroglyph_count=site.petroglyph_count,
            latitude=site.latitude,
            longitude=site.longitude,
        )
    for edge in edges:
        name_a = id_to_name.get(edge.site_a_id)
        name_b = id_to_name.get(edge.site_b_id)
        if name_a and name_b:
            graph.load_persisted_edge(
                name_a,
                name_b,
                weight=edge.weight,
                evidence_count=edge.evidence_count,
                shared_taxonomies=list(edge.shared_taxonomies or []),
            )

    return graph


@app.get("/graph", tags=["Grafo Social"])
async def get_graph(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Retorna el grafo de similitud iconográfica en formato JSON.
    Incluye nodos (sitios), aristas (similitudes) y métricas del resumen.
    """
    graph = await _build_graph_from_db(session)
    return graph.to_dict()


@app.get("/graph/export", tags=["Grafo Social"])
async def export_graph_html(session: AsyncSession = Depends(get_session)) -> FileResponse:
    """
    Exporta una visualización interactiva del grafo como HTML (PyVis).
    Descarga directa del archivo HTML con física de partículas y tooltips.
    """
    graph = await _build_graph_from_db(session)
    html_path = graph.export_html()
    if not html_path or not Path(html_path).exists():
        raise HTTPException(status_code=500, detail="Error generando el grafo HTML.")
    return FileResponse(html_path, media_type="text/html", filename="red_rupestre.html")


@app.get("/graph/export/image", tags=["Grafo Social"])
async def export_graph_image(session: AsyncSession = Depends(get_session)) -> FileResponse:
    """Exporta una imagen PNG estática del grafo social."""
    graph = await _build_graph_from_db(session)
    image_path = graph.export_image()
    if not image_path or not Path(image_path).exists():
        raise HTTPException(status_code=500, detail="Error generando la imagen del grafo.")
    return FileResponse(image_path, media_type="image/png", filename="red_rupestre.png")


@app.get("/graph/pagerank", tags=["Grafo Social"])
async def get_graph_pagerank(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Retorna el ranking de PageRank de los sitios en la red iconográfica.
    Sitios con mayor score son los más centrales e influyentes en la red.
    """
    graph = await _build_graph_from_db(session)
    pr = graph.pagerank()
    if not pr:
        return {"pagerank": {}, "top_site": None, "message": "Grafo sin datos suficientes"}
    sorted_pr = sorted(pr.items(), key=lambda x: x[1], reverse=True)
    return {
        "pagerank": {site: round(score, 6) for site, score in sorted_pr},
        "top_site": sorted_pr[0][0],
    }


@app.get("/graph/communities", tags=["Grafo Social"])
async def get_graph_communities(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Detecta comunidades iconográficas usando el algoritmo de Louvain.
    Cada comunidad agrupa sitios con alta similitud estilística entre sí.
    """
    graph = await _build_graph_from_db(session)
    communities = graph.communities()
    return {
        "communities": [sorted(list(c)) for c in communities],
        "count": len(communities),
    }


@app.get("/graph/betweenness", tags=["Grafo Social"])
async def get_graph_betweenness(session: AsyncSession = Depends(get_session)) -> dict:
    """
    Centralidad de intermediación: identifica sitios que actúan como puentes
    iconográficos entre distintas regiones o tradiciones rupestres.
    """
    graph = await _build_graph_from_db(session)
    bc = graph.betweenness_centrality()
    if not bc:
        return {"betweenness": {}, "top_bridge_site": None, "message": "Grafo sin aristas"}
    sorted_bc = sorted(bc.items(), key=lambda x: x[1], reverse=True)
    return {
        "betweenness": {site: round(score, 6) for site, score in sorted_bc},
        "top_bridge_site": sorted_bc[0][0],
    }


@app.get("/graph/sites/{site_id}/similar", tags=["Grafo Social"])
async def get_similar_sites(
    site_id: str,
    top_k: int = 5,
    session: AsyncSession = Depends(get_session),
) -> dict:
    """
    Retorna los sitios más similares iconográficamente a un sitio dado (por UUID).
    Útil para explorar tradiciones rupestres comparadas.
    """
    from infrastructure.database.models.models import RupestranSiteModel

    site_result = await session.execute(
        select(RupestranSiteModel).where(RupestranSiteModel.id == site_id)
    )
    site = site_result.scalar_one_or_none()
    if not site:
        raise HTTPException(status_code=404, detail=f"Sitio {site_id} no encontrado.")

    graph = await _build_graph_from_db(session)
    similar = graph.most_similar_sites(site.name, top_k=top_k)
    return {
        "site_id": site_id,
        "site_name": site.name,
        "similar_sites": similar,
    }


# ── Fichas ICANH ───────────────────────────────────────────────────────────────

@app.get("/fichas/{task_id}/pdf", tags=["Fichas ICANH"])
async def download_ficha_pdf(task_id: str) -> FileResponse:
    """Descarga la ficha ICANH en PDF para una tarea de clasificación dada."""
    pdf_path = Path(f"storage/fichas_icanh/{task_id}_ficha.pdf")
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="Ficha PDF no encontrada.")
    return FileResponse(str(pdf_path), media_type="application/pdf",
                        filename=f"ficha_icanh_{task_id}.pdf")


@app.get("/fichas/{task_id}/json", tags=["Fichas ICANH"])
async def download_ficha_json(task_id: str) -> dict:
    """Retorna la ficha ICANH en formato JSON para una tarea de clasificación."""
    import json
    json_path = Path(f"storage/fichas_icanh/{task_id}_ficha.json")
    if not json_path.exists():
        raise HTTPException(status_code=404, detail="Ficha JSON no encontrada.")
    return json.loads(json_path.read_text(encoding="utf-8"))
