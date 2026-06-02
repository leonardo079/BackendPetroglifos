"""A4 — Analista Cultural (RAG + Gemini): núcleo del módulo LLM."""
from __future__ import annotations

import json
import time
from pathlib import Path

import structlog
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sqlalchemy import text

from agents.base_agent import AgentInput, AgentOutput, BaseAgent
from adapters.outbound.embeddings.gemini_embedding_adapter import GeminiEmbeddingAdapter
from adapters.outbound.llm.gemini_adapter import GeminiAdapter
from config.settings import settings
from core.domain.enums.taxonomy import TaxonomyCategory
from core.domain.value_objects.classification_result import ClassificationResult
from rag.retrieval.retriever import RAGRetriever

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
        embedder: GeminiEmbeddingAdapter,
        session=None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._llm = llm
        self._retriever = retriever
        self._embedder = embedder
        self._session = session

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        payload = input.payload
        petroglyph_id: str = payload.get("petroglyph_id", input.task_id)
        motif_description: str = payload.get("motif_description", "")
        detected_shapes: list[str] = payload.get("detected_shapes", [])
        similarity_matches: list[dict] = payload.get("similarity_matches", [])
        site_metadata: dict = payload.get("site_metadata", {})

        try:
            taxonomy_hint = ""
            if similarity_matches:
                tax_counts: dict[str, int] = {}
                for m in similarity_matches:
                    t = str(m.get("taxonomy", "")).strip()
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

            prompt = self._build_prompt(
                retrieved_chunks=chunks,
                motif_description=motif_description,
                detected_shapes=detected_shapes,
                similarity_matches=similarity_matches,
            )

            fallback_used = False
            try:
                response_data = await self._llm.generate_json(prompt, system=_SYSTEM_PROMPT)
                taxonomy_raw = str(response_data.get("taxonomy", "Indeterminado"))
                taxonomy = TaxonomyCategory.from_str(taxonomy_raw).value
                confidence = float(response_data.get("confidence", 0.0))
                justification = str(response_data.get("justification", ""))
            except Exception as llm_err:
                fallback_used = True
                log.warning("a4_llm_fallback", error=str(llm_err), task_id=input.task_id)
                heuristic = self._heuristic_classification(
                    motif_description=motif_description,
                    detected_shapes=detected_shapes,
                    similarity_matches=similarity_matches,
                )
                taxonomy = str(heuristic["taxonomy"])
                confidence = float(heuristic["confidence"])
                justification = str(heuristic["justification"])

            result = ClassificationResult(
                taxonomy=taxonomy,
                confidence=confidence,
                justification=justification,
                retrieved_context=chunks,
                requires_validation=confidence < settings.confidence_threshold,
                low_context_quality=low_context,
                status="fallback" if fallback_used else "success",
            )

            if fallback_used:
                description_payload = self._heuristic_description(
                    taxonomy=taxonomy,
                    confidence=confidence,
                    justification=justification,
                    motif_description=motif_description,
                    detected_shapes=detected_shapes,
                    similarity_matches=similarity_matches,
                    site_metadata=site_metadata,
                )
            else:
                try:
                    description_payload = await self._generate_petroglyph_description(
                        taxonomy=taxonomy,
                        confidence=confidence,
                        justification=justification,
                        motif_description=motif_description,
                        detected_shapes=detected_shapes,
                        similarity_matches=similarity_matches,
                        site_metadata=site_metadata,
                    )
                except Exception as desc_err:
                    fallback_used = True
                    log.warning("a4_description_fallback", error=str(desc_err), task_id=input.task_id)
                    description_payload = self._heuristic_description(
                        taxonomy=taxonomy,
                        confidence=confidence,
                        justification=justification,
                        motif_description=motif_description,
                        detected_shapes=detected_shapes,
                        similarity_matches=similarity_matches,
                        site_metadata=site_metadata,
                    )

            detailed_description = str(description_payload.get("detailed_description", "")).strip()
            probable_site = str(description_payload.get("probable_site", "")).strip()
            site_probability = float(description_payload.get("site_probability", 0.0))
            key_figure_info = description_payload.get("key_figure_info", [])
            if not isinstance(key_figure_info, list):
                key_figure_info = [str(key_figure_info)]

            description_embedding = await self._embedder.embed(
                detailed_description or justification or motif_description
            )
            rag_feedback = await self._compute_rag_feedback(
                description_embedding=description_embedding,
                retrieved_chunks=chunks,
            )

            elapsed = round((time.monotonic() - t0) * 1000)

            if self._session:
                try:
                    await self._persist(
                        petroglyph_id=petroglyph_id,
                        result=result,
                        prompt=prompt,
                        latency_ms=elapsed,
                        detailed_description=detailed_description,
                        probable_site=probable_site,
                        site_probability=site_probability,
                        key_figure_info=key_figure_info,
                        description_embedding=description_embedding,
                        rag_feedback=rag_feedback,
                    )
                except Exception as persist_err:
                    log.warning("a4_persist_failed", error=str(persist_err), task_id=input.task_id)

            log.info(
                "a4_analyst_done",
                task_id=input.task_id,
                taxonomy=taxonomy,
                confidence=confidence,
                chunks_used=len(chunks),
                latency_ms=elapsed,
                fallback_used=fallback_used,
            )

            output_data = result.model_dump()
            output_data["petroglyph_description"] = {
                "detailed_description": detailed_description,
                "probable_site": probable_site,
                "site_probability": site_probability,
                "key_figure_info": key_figure_info,
            }
            output_data["rag_feedback"] = rag_feedback

            return AgentOutput(
                task_id=input.task_id,
                agent_name=self.name,
                result=output_data,
                status="success",
                metadata={
                    "latency_ms": elapsed,
                    "chunks_used": len(chunks),
                    "low_context": low_context,
                    "fallback_used": fallback_used,
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

    def _build_description_prompt(
        self,
        taxonomy: str,
        confidence: float,
        justification: str,
        motif_description: str,
        detected_shapes: list[str],
        similarity_matches: list[dict],
        site_metadata: dict,
    ) -> str:
        template = _jinja_env.get_template("templates/petroglyph_description_prompt.jinja2")
        return template.render(
            taxonomy=taxonomy,
            confidence=confidence,
            justification=justification,
            motif_description=motif_description,
            detected_shapes=detected_shapes,
            similarity_matches=similarity_matches,
            site_metadata=site_metadata,
        )

    async def _generate_petroglyph_description(
        self,
        taxonomy: str,
        confidence: float,
        justification: str,
        motif_description: str,
        detected_shapes: list[str],
        similarity_matches: list[dict],
        site_metadata: dict,
    ) -> dict:
        prompt = self._build_description_prompt(
            taxonomy=taxonomy,
            confidence=confidence,
            justification=justification,
            motif_description=motif_description,
            detected_shapes=detected_shapes,
            similarity_matches=similarity_matches,
            site_metadata=site_metadata,
        )
        description_data = await self._llm.generate_json(prompt, system=_SYSTEM_PROMPT)
        return {
            "detailed_description": description_data.get("detailed_description", ""),
            "probable_site": description_data.get("probable_site", ""),
            "site_probability": max(0.0, min(float(description_data.get("site_probability", 0.0)), 1.0)),
            "key_figure_info": description_data.get("key_figure_info", []),
        }

    def _heuristic_classification(
        self,
        *,
        motif_description: str,
        detected_shapes: list[str],
        similarity_matches: list[dict],
    ) -> dict:
        scores = {cat.value: 0 for cat in TaxonomyCategory}
        evidence: list[str] = []

        taxonomy_votes: dict[str, int] = {}
        for match in similarity_matches:
            taxonomy = str(match.get("taxonomy", "")).strip()
            if taxonomy:
                taxonomy_votes[taxonomy] = taxonomy_votes.get(taxonomy, 0) + 1
        if taxonomy_votes:
            best_taxonomy = max(taxonomy_votes, key=taxonomy_votes.get)
            category = TaxonomyCategory.from_str(best_taxonomy)
            if category != TaxonomyCategory.INDETERMINADO:
                scores[category.value] += 3
                evidence.append(f"similitud iconográfica dominante: {best_taxonomy}")

        text = f"{motif_description} {' '.join(detected_shapes)}".lower()
        if any(word in text for word in ("figura humana", "humano", "rostro", "persona", "brazos", "cuerpo")):
            scores[TaxonomyCategory.ANTROPOMORFO.value] += 3
            evidence.append("rasgos antropomorfos en la descripción o las formas")
        if any(word in text for word in ("animal", "ave", "serpiente", "felino", "zoomorfo")):
            scores[TaxonomyCategory.ZOOMORFO.value] += 3
            evidence.append("rasgos zoomorfos en la descripción o las formas")
        if any(word in text for word in ("círculo", "circulo", "línea", "linea", "espiral", "triángulo", "triangulo", "cuadrado", "geométr", "geometr")):
            scores[TaxonomyCategory.GEOMETRICO.value] += 3
            evidence.append("patrones geométricos en la descripción o las formas")
        if any(word in text for word in ("sol", "luna", "estrella", "astro", "astron")):
            scores[TaxonomyCategory.ASTRONOMICO.value] += 3
            evidence.append("referencias astronómicas en la descripción")
        if any(word in text for word in ("planta", "flor", "hoja", "vegetal", "fitom")):
            scores[TaxonomyCategory.FITOMORFO.value] += 3
            evidence.append("rasgos fitomorfos en la descripción")

        if len(set(taxonomy_votes)) >= 2:
            scores[TaxonomyCategory.HIBRIDO.value] += 2
            evidence.append("coincidencias mixtas en el comparador visual")

        taxonomy = max(scores, key=scores.get)
        if scores[taxonomy] == 0:
            taxonomy = TaxonomyCategory.INDETERMINADO.value
            confidence = 0.25
            evidence.append("no hubo evidencia fuerte suficiente")
        else:
            confidence = min(0.75, 0.30 + (scores[taxonomy] * 0.08))

        return {
            "taxonomy": taxonomy,
            "confidence": round(confidence, 2),
            "justification": (
                "Clasificación heurística de respaldo. "
                + ("; ".join(evidence) if evidence else "Se priorizó la señal más consistente disponible.")
            ),
        }

    def _heuristic_description(
        self,
        *,
        taxonomy: str,
        confidence: float,
        justification: str,
        motif_description: str,
        detected_shapes: list[str],
        similarity_matches: list[dict],
        site_metadata: dict,
    ) -> dict:
        best_match = similarity_matches[0] if similarity_matches else {}
        probable_site = str(best_match.get("site_name", site_metadata.get("site", "No definido")))
        site_probability = float(best_match.get("similarity_score", 0.35 if not similarity_matches else 0.55))
        shape_text = ", ".join(detected_shapes) if detected_shapes else "sin formas detectadas"
        detailed_description = (
            f"Descripción heurística del petroglifo clasificado como {taxonomy}. "
            f"El sistema observó {shape_text} y el motivo fue descrito como: {motif_description or 'no disponible'}. "
            f"Justificación: {justification}"
        )
        key_figure_info = [
            f"Clasificación de respaldo: {taxonomy}",
            f"Confianza estimada: {round(confidence * 100, 1)}%",
            f"Formas observadas: {shape_text}",
        ]
        return {
            "detailed_description": detailed_description,
            "probable_site": probable_site,
            "site_probability": max(0.0, min(site_probability, 1.0)),
            "key_figure_info": key_figure_info,
        }

    async def _compute_rag_feedback(
        self,
        description_embedding: list[float],
        retrieved_chunks: list[dict],
    ) -> dict:
        if not self._session or not retrieved_chunks:
            return {"avg_similarity": 0.0, "top_matches": []}

        top_matches: list[dict] = []
        for chunk in retrieved_chunks:
            chunk_id = chunk.get("id")
            if not chunk_id:
                continue
            similarity_sql = text("""
                SELECT
                    source_document,
                    chunk_text,
                    1 - (embedding <=> CAST(:desc_vec AS vector)) AS similarity
                FROM archaeological_chunks
                WHERE id = CAST(:chunk_id AS uuid)
                LIMIT 1
            """)
            row = (
                await self._session.execute(
                    similarity_sql,
                    {"desc_vec": str(description_embedding), "chunk_id": chunk_id},
                )
            ).first()
            if not row:
                continue
            top_matches.append(
                {
                    "source": row.source_document,
                    "text": row.chunk_text,
                    "similarity": float(row.similarity),
                }
            )

        if not top_matches:
            return {"avg_similarity": 0.0, "top_matches": []}

        sorted_matches = sorted(top_matches, key=lambda m: m["similarity"], reverse=True)
        avg_similarity = sum(m["similarity"] for m in sorted_matches) / len(sorted_matches)
        return {
            "avg_similarity": round(avg_similarity, 4),
            "top_matches": sorted_matches[:3],
            "consistency_label": "alta" if avg_similarity >= 0.75 else "media" if avg_similarity >= 0.6 else "baja",
        }

    async def _persist(
        self,
        petroglyph_id: str,
        result: ClassificationResult,
        prompt: str,
        latency_ms: int,
        detailed_description: str,
        probable_site: str,
        site_probability: float,
        key_figure_info: list,
        description_embedding: list[float],
        rag_feedback: dict,
    ) -> None:
        from infrastructure.database.models.models import (
            LLMClassification,
            PetroglyphDescriptionEmbedding,
            PetroglyphModel,
            PromptLog,
        )

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
        petroglyph_desc = PetroglyphDescriptionEmbedding(
            petroglyph_id=petroglyph_id,
            taxonomy=result.taxonomy,
            detailed_description=detailed_description,
            probable_site=probable_site,
            site_probability=site_probability,
            key_figure_info=key_figure_info,
            embedding=description_embedding,
            rag_feedback=rag_feedback,
        )
        prompt_log = PromptLog(
            petroglyph_id=petroglyph_id,
            prompt=prompt[:10000],
            response=json.dumps(result.model_dump()),
            latency_ms=latency_ms,
            status_code="ok",
        )
        self._session.add(classification)
        self._session.add(petroglyph_desc)
        self._session.add(prompt_log)
        await self._session.commit()
