"""A3 — Comparador iconográfico (EfficientNet-B0 + pgvector) + Grafo social."""
from __future__ import annotations
import time
import os
import structlog
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from adapters.outbound.vector_store.pgvector_adapter import ImageVectorAdapter
from graphs.social_graph import PetroglyphSocialGraph
from core.domain.site_normalization import canonicalize_municipality, canonicalize_site_name

log = structlog.get_logger(__name__)

_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_efficientnet():
    """Carga EfficientNet-B0 preentrenado como extractor de features."""
    try:
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=0)
        model.eval()
        return model
    except Exception as e:
        log.warning("efficientnet_load_failed", error=str(e))
        return None


_MODEL = _load_efficientnet()


def extract_image_embedding(image_path: str) -> list[float] | None:
    """Extrae embedding de 1280 dims con EfficientNet-B0."""
    if _MODEL is None or not os.path.exists(image_path):
        return None
    try:
        img = Image.open(image_path).convert("RGB")
        tensor = _TRANSFORM(img).unsqueeze(0)
        with torch.no_grad():
            features = _MODEL(tensor)
        return features.squeeze().numpy().tolist()
    except Exception as e:
        log.error("embedding_extraction_error", path=image_path, error=str(e))
        return None


class ComparatorAgent(BaseAgent):
    name = "a3_comparator"

    def __init__(
        self,
        image_vector_adapter: ImageVectorAdapter | None = None,
        social_graph: PetroglyphSocialGraph | None = None,
        session=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._image_vector = image_vector_adapter
        self._social_graph = social_graph
        self._session = session

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("preprocessed_image_path", "") or \
                          input.payload.get("image_path", "")
        current_site: str = canonicalize_site_name(input.payload.get("site", ""))
        current_municipality: str = canonicalize_municipality(input.payload.get("municipality", ""))
        site_id: str = input.payload.get("site_id", "")

        matches: list[dict] = []

        # 1. Extraer embedding de la imagen actual
        embedding = extract_image_embedding(image_path)

        if embedding and self._image_vector:
            # 2. Buscar similitudes en el corpus de referencia
            raw_matches = await self._image_vector.similarity_search(
                query_vector=embedding, k=5, min_similarity=0.60
            )
            matches = [
                {
                    "site_name": canonicalize_site_name(m["site_name"]),
                    "municipality": canonicalize_municipality(m["municipality"]),
                    "reference_name": m["reference_name"],
                    "taxonomy": m["taxonomy"],
                    "similarity_score": round(m["similarity_score"], 4),
                    "image_path": m["image_path"],
                }
                for m in raw_matches
            ]

            # 3. Actualizar el grafo en memoria usando nombres de sitio como IDs de nodo.
            # Esto garantiza consistencia con la reconstrucción del grafo desde la BD.
            node_a = current_site
            if self._social_graph and node_a and matches:
                for match in matches:
                    node_b = canonicalize_site_name(match.get("site_name", ""))
                    if node_b and match["similarity_score"] >= 0.70:
                        self._social_graph.add_or_update_edge(
                            site_a=node_a,
                            site_b=node_b,
                            weight=match["similarity_score"],
                            taxonomy=match.get("taxonomy", ""),
                        )

            # 4. Persistir aristas en la tabla site_graph_edges (si hay sesión disponible).
            if self._session and (current_site or site_id) and matches:
                await self._persist_edges(
                    current_site_name=current_site,
                    current_site_id=site_id,
                    current_municipality=current_municipality,
                    matches=matches,
                )

        edges_persisted = bool(self._session and matches)
        elapsed = round((time.monotonic() - t0) * 1000)
        log.info(
            "a3_comparator_done",
            task_id=input.task_id,
            matches=len(matches),
            edges_persisted=edges_persisted,
            latency_ms=elapsed,
        )

        return AgentOutput(
            task_id=input.task_id,
            agent_name=self.name,
            result={"similarity_matches": matches},
            status="success",
            metadata={
                "latency_ms": elapsed,
                "embedding_available": embedding is not None,
                "graph_updated": self._social_graph is not None and len(matches) > 0,
                "edges_persisted": edges_persisted,
            },
        )

    # ── Persistencia en BD ─────────────────────────────────────────────────────

    async def _get_or_create_site(self, name: str, municipality: str = "") -> str | None:
        """
        Retorna el UUID del sitio que tenga ese nombre en rupestrian_sites.
        Si no existe, lo crea con los datos disponibles y hace flush.
        Maneja la condición de carrera (múltiples workers Celery) capturando
        IntegrityError y releyendo el registro creado por el worker concurrente.
        """
        from sqlalchemy import select
        from sqlalchemy.exc import IntegrityError
        from infrastructure.database.models.models import RupestranSiteModel

        name = canonicalize_site_name(name)
        municipality = canonicalize_municipality(municipality)
        result = await self._session.execute(
            select(RupestranSiteModel).where(RupestranSiteModel.name == name).limit(1)
        )
        site = result.scalar_one_or_none()
        if site:
            return site.id

        try:
            new_site = RupestranSiteModel(name=name, municipality=municipality)
            self._session.add(new_site)
            await self._session.flush()  # obtener el ID generado sin hacer commit
            log.debug("a3_site_auto_created", name=name)
            return new_site.id
        except IntegrityError:
            # Condición de carrera: otro worker creó el sitio entre nuestro SELECT y el INSERT.
            # Revertimos el savepoint de flush y releemos el registro ya existente.
            await self._session.rollback()
            result = await self._session.execute(
                select(RupestranSiteModel).where(RupestranSiteModel.name == name).limit(1)
            )
            existing = result.scalar_one_or_none()
            log.debug("a3_site_race_resolved", name=name, found=existing is not None)
            return existing.id if existing else None

    async def _persist_edges(
        self,
        current_site_name: str,
        current_site_id: str,
        current_municipality: str,
        matches: list[dict],
    ) -> None:
        """
        Persiste las aristas de similitud (score >= 0.70) en site_graph_edges.

        - Si el sitio no existe en rupestrian_sites, lo crea automáticamente.
        - Normaliza el orden de los UUIDs para cumplir la restricción UNIQUE.
        - Usa flush() en lugar de commit() para no interferir con la transacción
          del pipeline (el commit lo hace A4 o el contexto de la tarea Celery).
        """
        from sqlalchemy import select, and_
        from infrastructure.database.models.models import RupestranSiteModel, SiteGraphEdge

        try:
            current_site_name = canonicalize_site_name(current_site_name)
            current_municipality = canonicalize_municipality(current_municipality)
            # Resolver UUID del sitio que se está analizando
            if current_site_id and current_site_id.count("-") == 4:
                site_a_uuid = current_site_id
            else:
                site_a_uuid = await self._get_or_create_site(
                    current_site_name, municipality=current_municipality
                )
            if not site_a_uuid:
                return

            persisted = 0
            for match in matches:
                score = match.get("similarity_score", 0.0)
                if score < 0.70:
                    continue

                match_name = match.get("site_name", "")
                if not match_name:
                    continue

                site_b_uuid = await self._get_or_create_site(
                    canonicalize_site_name(match_name),
                    municipality=match.get("municipality", ""),
                )
                if not site_b_uuid or site_a_uuid == site_b_uuid:
                    continue

                taxonomy = match.get("taxonomy", "")

                # Normalizar orden para la restricción UNIQUE(site_a_id, site_b_id)
                id_a, id_b = sorted([site_a_uuid, site_b_uuid])

                existing = (await self._session.execute(
                    select(SiteGraphEdge).where(
                        and_(
                            SiteGraphEdge.site_a_id == id_a,
                            SiteGraphEdge.site_b_id == id_b,
                        )
                    )
                )).scalar_one_or_none()

                if existing:
                    # Promedio acumulativo del peso
                    n = existing.evidence_count
                    existing.weight = round((existing.weight * n + score) / (n + 1), 4)
                    existing.evidence_count = n + 1
                    current_taxonomies = list(existing.shared_taxonomies or [])
                    if taxonomy and taxonomy not in current_taxonomies:
                        existing.shared_taxonomies = current_taxonomies + [taxonomy]
                else:
                    self._session.add(SiteGraphEdge(
                        site_a_id=id_a,
                        site_b_id=id_b,
                        weight=round(score, 4),
                        shared_taxonomies=[taxonomy] if taxonomy else [],
                        evidence_count=1,
                    ))
                persisted += 1

            await self._session.flush()
            log.info(
                "a3_graph_edges_persisted",
                site=current_site_name,
                edges_upserted=persisted,
            )
        except Exception as exc:
            log.error("a3_graph_persist_error", site=current_site_name, error=str(exc))
