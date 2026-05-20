"""
Orquestador LangGraph del sistema de petroglifos.

Flujo de ejecución:
    START → A1 (preprocesar) → A2 (detectar) → [router] →
    ├─ motivos visibles → A3 (comparar) → A4 (analizar) → A6 (documentar) → END
    └─ deterioro → A5 (reconstruir) → A2 (detectar) → A4 → A6 → END
"""
from __future__ import annotations
import time
import structlog
from typing import Literal
from langgraph.graph import StateGraph, END
from orchestrator.state.graph_state import PetroglyphState
from agents.base_agent import AgentInput

log = structlog.get_logger(__name__)


class PetroglyphOrchestrator:
    """
    Orquestador principal del sistema multiagente.
    
    Uso:
        orchestrator = PetroglyphOrchestrator(a1, a2, a3, a4, a5, a6, social_graph)
        result = await orchestrator.run(task_id, image_path, site_metadata)
    """

    def __init__(
        self,
        a1_preprocessor,
        a2_detector,
        a3_comparator,
        a4_analyst,
        a5_reconstructor,
        a6_documentor,
        social_graph=None,
    ) -> None:
        self._a1 = a1_preprocessor
        self._a2 = a2_detector
        self._a3 = a3_comparator
        self._a4 = a4_analyst
        self._a5 = a5_reconstructor
        self._a6 = a6_documentor
        self._social_graph = social_graph

        # Construir el grafo
        self._graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Construye el grafo de ejecución con LangGraph."""
        workflow = StateGraph(PetroglyphState)

        # Nodos
        workflow.add_node("a1_preprocessor", self._run_a1)
        workflow.add_node("a2_detector", self._run_a2)
        workflow.add_node("a3_comparator", self._run_a3)
        workflow.add_node("a4_analyst", self._run_a4)
        workflow.add_node("a5_reconstructor", self._run_a5)
        workflow.add_node("a6_documentor", self._run_a6)

        # Flujo lineal inicial
        workflow.set_entry_point("a1_preprocessor")
        workflow.add_edge("a1_preprocessor", "a2_detector")

        # Router condicional después de A2
        workflow.add_conditional_edges(
            "a2_detector",
            self._route_after_detection,
            {
                "a3_comparator": "a3_comparator",
                "a5_reconstructor": "a5_reconstructor",
            },
        )

        # Flujo con motivos visibles: A3 → A4 → A6 → END
        workflow.add_edge("a3_comparator", "a4_analyst")

        # Flujo de reconstrucción: A5 → A2 (re-detectar) → A4 → A6 → END
        workflow.add_edge("a5_reconstructor", "a2_detector")

        # Convergencia final: A4 → A6 → END
        workflow.add_edge("a4_analyst", "a6_documentor")
        workflow.add_edge("a6_documentor", END)

        return workflow.compile()

    # ── Funciones de ejecución de nodos ───────────────────────────────────────

    async def _run_a1(self, state: PetroglyphState) -> PetroglyphState:
        """A1: Preprocesar imagen."""
        log.info("langgraph_node", node="a1_preprocessor", task_id=state["petroglyph_id"])
        
        agent_input = AgentInput(
            task_id=state["petroglyph_id"],
            payload={"image_path": state["raw_image_path"]},
        )
        result = await self._a1.run(agent_input)
        
        state["preprocessed_image_path"] = result.result.get("preprocessed_image_path", "")
        return state

    async def _run_a2(self, state: PetroglyphState) -> PetroglyphState:
        """A2: Detectar motivos."""
        log.info("langgraph_node", node="a2_detector", task_id=state["petroglyph_id"])
        
        # Determinar qué imagen usar (reconstruida si existe, sino preprocesada)
        image_path = state.get("reconstructed_image_path") or state.get("preprocessed_image_path", "")
        
        agent_input = AgentInput(
            task_id=state["petroglyph_id"],
            payload={
                "image_path": image_path,
                "preprocessed_image_path": state.get("preprocessed_image_path", ""),
            },
        )
        result = await self._a2.run(agent_input)
        
        state["motif_description"] = result.result.get("motif_description", "")
        state["detected_shapes"] = result.result.get("detected_shapes", [])
        state["motifs_visible"] = result.result.get("motifs_visible", False)
        state["detection_confidence"] = result.result.get("detection_confidence", 0.0)
        
        # Guardar info de deterioro para el router
        state["_deterioration_detected"] = result.result.get("deterioration_detected", False)
        
        return state

    async def _run_a3(self, state: PetroglyphState) -> PetroglyphState:
        """A3: Comparar iconográficamente."""
        log.info("langgraph_node", node="a3_comparator", task_id=state["petroglyph_id"])

        # Usar la imagen reconstruida si existe (flujo post-A5), sino la preprocesada
        best_image = (
            state.get("reconstructed_image_path")
            or state.get("preprocessed_image_path", "")
        )

        agent_input = AgentInput(
            task_id=state["petroglyph_id"],
            payload={
                "preprocessed_image_path": best_image,
                "site": state["site_metadata"].get("site", ""),
                "municipality": state["site_metadata"].get("municipality", ""),
                "site_id": state["site_metadata"].get("site_id", ""),
            },
        )
        result = await self._a3.run(agent_input)
        
        state["similarity_matches"] = result.result.get("similarity_matches", [])
        return state

    async def _run_a4(self, state: PetroglyphState) -> PetroglyphState:
        """A4: Análisis cultural con RAG."""
        log.info("langgraph_node", node="a4_analyst", task_id=state["petroglyph_id"])
        
        agent_input = AgentInput(
            task_id=state["petroglyph_id"],
            payload={
                "petroglyph_id": state["petroglyph_id"],
                "motif_description": state.get("motif_description", ""),
                "detected_shapes": state.get("detected_shapes", []),
                "similarity_matches": state.get("similarity_matches", []),
            },
        )
        result = await self._a4.run(agent_input)
        
        state["a4_taxonomy_result"] = result.result
        state["a4_requires_validation"] = result.result.get("requires_validation", True)
        return state

    async def _run_a5(self, state: PetroglyphState) -> PetroglyphState:
        """A5: Reconstruir con GAN."""
        log.info("langgraph_node", node="a5_reconstructor", task_id=state["petroglyph_id"])
        
        agent_input = AgentInput(
            task_id=state["petroglyph_id"],
            payload={
                "preprocessed_image_path": state.get("preprocessed_image_path", ""),
                "deterioration_detected": state.get("_deterioration_detected", True),
            },
        )
        result = await self._a5.run(agent_input)
        
        state["reconstructed_image_path"] = result.result.get("reconstructed_image_path", "")
        return state

    async def _run_a6(self, state: PetroglyphState) -> PetroglyphState:
        """A6: Generar ficha ICANH."""
        log.info("langgraph_node", node="a6_documentor", task_id=state["petroglyph_id"])
        
        # Consolidar payload con todos los resultados
        payload = {
            "petroglyph_id": state["petroglyph_id"],
            "site": state["site_metadata"].get("site", ""),
            "municipality": state["site_metadata"].get("municipality", ""),
            "department": state["site_metadata"].get("department", ""),
            "gps_coordinates": state["site_metadata"].get("gps_coordinates", {}),
            "image_path": state.get("raw_image_path", ""),
            "preprocessed_image_path": state.get("preprocessed_image_path", ""),
            "reconstructed_image_path": state.get("reconstructed_image_path", ""),
            "motif_description": state.get("motif_description", ""),
            "detected_shapes": state.get("detected_shapes", []),
            "similarity_matches": state.get("similarity_matches", []),
            "taxonomy": state["a4_taxonomy_result"].get("taxonomy", "Indeterminado"),
            "confidence": state["a4_taxonomy_result"].get("confidence", 0.0),
            "justification": state["a4_taxonomy_result"].get("justification", ""),
            "requires_validation": state.get("a4_requires_validation", True),
            "conservation_status": state["site_metadata"].get("conservation_status", "Regular"),
            "researcher_notes": state["site_metadata"].get("researcher_notes", ""),
        }
        
        agent_input = AgentInput(task_id=state["petroglyph_id"], payload=payload)
        result = await self._a6.run(agent_input)
        
        state["icanh_pdf_url"] = result.result.get("icanh_pdf_url", "")
        state["icanh_json"] = result.result.get("icanh_record", {})
        return state

    # ── Router condicional ────────────────────────────────────────────────────

    def _route_after_detection(self, state: PetroglyphState) -> Literal["a3_comparator", "a5_reconstructor"]:
        """
        Router condicional post-A2.

        Lógica:
        - Si la imagen ya fue reconstruida por A5 → A3 (comparación con corpus)
        - Si hay motivos visibles y sin deterioro severo → A3
        - Si hay deterioro o no hay motivos visibles → A5 (reconstrucción)
        """
        motifs_visible = state.get("motifs_visible", False)
        deterioration = state.get("_deterioration_detected", False)

        # Post-reconstrucción: la imagen ya fue restaurada por A5,
        # continuar con comparación iconográfica sobre la imagen reconstruida.
        if state.get("reconstructed_image_path"):
            log.info("router_decision", route="a3_comparator", reason="post_reconstruction")
            return "a3_comparator"

        if motifs_visible and not deterioration:
            log.info("router_decision", route="a3_comparator", reason="motifs_visible")
            return "a3_comparator"

        log.info("router_decision", route="a5_reconstructor", reason="needs_reconstruction")
        return "a5_reconstructor"

    # ── Método principal de ejecución ─────────────────────────────────────────

    async def run(
        self,
        task_id: str,
        raw_image_path: str,
        site_metadata: dict,
    ) -> dict:
        """
        Ejecuta el pipeline completo.
        
        Args:
            task_id: ID único de la tarea
            raw_image_path: Ruta a la imagen original
            site_metadata: Metadata del sitio (nombre, coordenadas, etc.)
        
        Returns:
            Estado final con todos los resultados
        """
        t0 = time.monotonic()
        log.info("orchestrator_start", task_id=task_id, image=raw_image_path)

        # Estado inicial
        initial_state: PetroglyphState = {
            "petroglyph_id": task_id,
            "raw_image_path": raw_image_path,
            "site_metadata": site_metadata,
        }

        try:
            # Ejecutar el grafo
            final_state = await self._graph.ainvoke(initial_state)
            
            elapsed = round((time.monotonic() - t0) * 1000)
            log.info("orchestrator_complete",
                     task_id=task_id,
                     total_latency_ms=elapsed,
                     taxonomy=final_state["a4_taxonomy_result"].get("taxonomy"),
                     confidence=final_state["a4_taxonomy_result"].get("confidence"))

            return {
                "task_id": task_id,
                "status": "success",
                "total_time_ms": elapsed,
                "classification": final_state["a4_taxonomy_result"],
                "icanh_pdf_url": final_state.get("icanh_pdf_url", ""),
                "icanh_json": final_state.get("icanh_json", {}),
            }

        except Exception as e:
            elapsed = round((time.monotonic() - t0) * 1000)
            log.error("orchestrator_error", task_id=task_id, error=str(e), latency_ms=elapsed)
            return {
                "task_id": task_id,
                "status": "error",
                "error": str(e),
                "total_time_ms": elapsed,
            }


# ── Factory para instanciar el orquestador con dependencias ──────────────────

async def create_orchestrator(session) -> PetroglyphOrchestrator:
    """
    Factory para crear el orquestador con todas las dependencias inyectadas.
    
    Uso en FastAPI:
        @app.post("/classify")
        async def classify(session: AsyncSession = Depends(get_session)):
            orchestrator = await create_orchestrator(session)
            result = await orchestrator.run(task_id, image_path, metadata)
    """
    from agents.a1_preprocessor.agent import PreprocessorAgent
    from agents.a2_detector.agent import DetectorAgent
    from agents.a3_comparator.agent import ComparatorAgent
    from agents.a4_analyst.agent import CulturalAnalystAgent
    from agents.a5_reconstructor.agent import ReconstructorAgent
    from agents.a6_documentor.agent import DocumentorAgent
    
    from adapters.outbound.llm.gemini_adapter import GeminiAdapter
    from adapters.outbound.embeddings.gemini_embedding_adapter import GeminiEmbeddingAdapter
    from adapters.outbound.vector_store.pgvector_adapter import PgvectorAdapter, ImageVectorAdapter
    from rag.retrieval.retriever import RAGRetriever
    from graphs.social_graph import PetroglyphSocialGraph

    # Adaptadores
    llm = GeminiAdapter(lite=False)
    embedder = GeminiEmbeddingAdapter()
    vector_store = PgvectorAdapter(session)
    image_vector = ImageVectorAdapter(session)
    retriever = RAGRetriever(embedder, vector_store)
    social_graph = PetroglyphSocialGraph()

    # Agentes
    a1 = PreprocessorAgent()
    a2 = DetectorAgent()
    a3 = ComparatorAgent(image_vector_adapter=image_vector, social_graph=social_graph, session=session)
    a4 = CulturalAnalystAgent(llm=llm, retriever=retriever, session=session)
    a5 = ReconstructorAgent()
    a6 = DocumentorAgent()

    return PetroglyphOrchestrator(a1, a2, a3, a4, a5, a6, social_graph)