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

    def __init__(self, image_vector_adapter: ImageVectorAdapter | None = None,
                 social_graph: PetroglyphSocialGraph | None = None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._image_vector = image_vector_adapter
        self._social_graph = social_graph

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("preprocessed_image_path", "") or \
                          input.payload.get("image_path", "")
        current_site: str = input.payload.get("site", "")
        current_municipality: str = input.payload.get("municipality", "")
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
                    "site_name": m["site_name"],
                    "municipality": m["municipality"],
                    "reference_name": m["reference_name"],
                    "taxonomy": m["taxonomy"],
                    "similarity_score": round(m["similarity_score"], 4),
                    "image_path": m["image_path"],
                }
                for m in raw_matches
            ]

            # 3. Actualizar el grafo social con las similitudes encontradas
            if self._social_graph and site_id and matches:
                for match in matches:
                    match_site_id = match.get("site_name", "")  # Usar site_name como ID temporal
                    if match_site_id and match["similarity_score"] >= 0.70:
                        self._social_graph.add_or_update_edge(
                            site_a=site_id or current_site,
                            site_b=match_site_id,
                            weight=match["similarity_score"],
                            taxonomy=match.get("taxonomy", ""),
                        )

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("a3_comparator_done",
                 task_id=input.task_id,
                 matches=len(matches),
                 latency_ms=elapsed)

        return AgentOutput(
            task_id=input.task_id,
            agent_name=self.name,
            result={"similarity_matches": matches},
            status="success",
            metadata={
                "latency_ms": elapsed,
                "embedding_available": embedding is not None,
                "graph_updated": self._social_graph is not None and len(matches) > 0,
            },
        )