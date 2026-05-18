"""A4 — Analista Cultural (RAG + Gemini): núcleo del módulo LLM."""
from __future__ import annotations
import json
import time
import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from pathlib import Path
from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from adapters.outbound.llm.gemini_adapter import GeminiAdapter
from rag.retrieval.retriever import RAGRetriever
from core.domain.enums.taxonomy import TaxonomyCategory
from core.domain.value_objects.classification_result import ClassificationResult
from config.settings import settings

log = structlog.get_logger(__name__)

_PROMPTS_DIR = Path("prompts")
_jinja_env = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=select_autoescape([]),
)

_SYSTEM_PROMPT = (_PROMPTS_DIR / "system" / "archaeological_analyst.txt").read_text(encoding="utf-8")


class CulturalAnalystAgent(BaseAgent):
    name = "a4_analyst"

    def __init__(
        self,
        llm: GeminiAdapter,
        retriever: RAGRetriever,
        session=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._retriever = retriever
        self._session = session

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        payload = input.payload
        petroglyph_id: str = payload.get("petroglyph_id", input.task_id)
        motif_description: str = payload.get("motif_description", "")
        detected_shapes: list[str] = payload.get("detected_shapes", [])
        similarity_matches: list[dict] = payload.get("similarity_matches", [])

        try:
            # 1. Recuperar contexto del corpus arqueológico.
            # Si A3 devuelvió matches con taxonomía dominante, usarla como hint
            # para que el retriever enriquezca la consulta y mejore el recall.
            taxonomy_hint = ""
            if similarity_matches:
                tax_counts: dict[str, int] = {}
                for m in similarity_matches:
                    t = m.get("taxonomy", "")
                    if t:
                        tax_counts[t] = tax_counts.get(t, 0) + 1
                if tax_counts:
                    taxonomy_hint = max(tax_counts, key=tax_counts.get)

            chunks = await self._retriever.retrieve_for_motif(
                motif_description=motif_description,
                detected_shapes=detected_shapes,
                taxonomy_hint=taxonomy_hint,
            )
            low_context = len(chunks) < 2

            # 2. Construir prompt con Jinja2
            prompt = self._build_prompt(
                retrieved_chunks=chunks,
                motif_description=motif_description,
                detected_shapes=detected_shapes,
                similarity_matches=similarity_matches,
            )

            # 3. Inferencia Gemini
            response_data = await self._llm.generate_json(prompt, system=_SYSTEM_PROMPT)

            # 4. Validar y normalizar taxonomía
            taxonomy_raw = response_data.get("taxonomy", "Indeterminado")
            taxonomy = TaxonomyCategory.from_str(taxonomy_raw).value
            confidence = float(response_data.get("confidence", 0.0))
            justification = response_data.get("justification", "")

            result = ClassificationResult(
                taxonomy=taxonomy,
                confidence=confidence,
                justification=justification,
                retrieved_context=chunks,
                requires_validation=confidence < settings.confidence_threshold,
                low_context_quality=low_context,
                status="success",
            )

            elapsed = round((time.monotonic() - t0) * 1000)

            # 5. Persistir en base de datos si hay sesión disponible
            if self._session:
                try:
                    await self._persist(petroglyph_id, result, prompt, elapsed)
                except Exception as persist_err:
                    log.warning(
                        "a4_persist_failed",
                        error=str(persist_err),
                        task_id=input.task_id,
                    )

            log.info("a4_analyst_done",
                     task_id=input.task_id,
                     taxonomy=taxonomy,
                     confidence=confidence,
                     chunks_used=len(chunks),
                     latency_ms=elapsed)

            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result=result.model_dump(),
                status="success",
                metadata={
                    "latency_ms": elapsed,
                    "chunks_used": len(chunks),
                    "low_context": low_context,
                },
            )

        except Exception as e:
            log.error("a4_analyst_error", error=str(e), task_id=input.task_id)
            fallback = ClassificationResult(
                taxonomy="Indeterminado",
                confidence=0.0,
                justification=f"Error en clasificación: {str(e)}",
                requires_validation=True,
                status="error",
            )
            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result=fallback.model_dump(),
                status="fallback",
                metadata={"error": str(e)},
            )

    def _build_prompt(
        self,
        retrieved_chunks: list[dict],
        motif_description: str,
        detected_shapes: list[str],
        similarity_matches: list[dict],
    ) -> str:
        template = _jinja_env.get_template("templates/classification_prompt.jinja2")
        return template.render(
            retrieved_chunks=retrieved_chunks,
            motif_description=motif_description,
            detected_shapes=detected_shapes,
            similarity_matches=similarity_matches,
            valid_categories=TaxonomyCategory.valid_values(),
        )

    async def _persist(
        self,
        petroglyph_id: str,
        result: ClassificationResult,
        prompt: str,
        latency_ms: int,
    ) -> None:
        from infrastructure.database.models.models import LLMClassification, PromptLog, PetroglyphModel

        # LLMClassification tiene FK NOT NULL hacia petroglyphs.id.
        # El pipeline no crea PetroglyphModel explícitamente (usa task_id como ID),
        # así que lo creamos como stub si aún no existe para satisfacer la FK.
        existing = await self._session.get(PetroglyphModel, petroglyph_id)
        if not existing:
            stub = PetroglyphModel(id=petroglyph_id)
            self._session.add(stub)
            await self._session.flush()
            log.debug("a4_petroglyph_stub_created", petroglyph_id=petroglyph_id)

        classification = LLMClassification(
            petroglyph_id=petroglyph_id,
            taxonomy=result.taxonomy,
            confidence=result.confidence,
            justification=result.justification,
            retrieved_context=result.retrieved_context,
            requires_validation=result.requires_validation,
            low_context_quality=result.low_context_quality,
            status=result.status,
        )
        prompt_log = PromptLog(
            petroglyph_id=petroglyph_id,
            prompt=prompt[:10000],
            response=json.dumps(result.model_dump()),
            latency_ms=latency_ms,
            status_code="ok",
        )
        self._session.add(classification)
        self._session.add(prompt_log)
        await self._session.commit()