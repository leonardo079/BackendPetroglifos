"""
Caso de uso: ClasificarPetroglifo.

Implementa el puerto de entrada ClassifyPetroglyphPort y coordina
el pipeline completo sin acoplarse a ningún adaptador concreto.
"""
from __future__ import annotations
import uuid
import structlog
from core.ports.inbound.classify_petroglyph_port import ClassifyPetroglyphPort
from core.domain.value_objects.classification_result import ClassificationResult

log = structlog.get_logger(__name__)


class ClassifyPetroglyphUseCase(ClassifyPetroglyphPort):
    """
    Caso de uso principal que orquesta la clasificación taxonómica.

    Diseñado para ser instanciado por el adaptador FastAPI (o CLI),
    recibe el orquestador LangGraph como dependencia.

    Uso:
        use_case = ClassifyPetroglyphUseCase(orchestrator)
        result = await use_case.classify({
            "image_path": "/path/to/img.jpg",
            "site": "Villa de Leyva",
            "municipality": "Villa de Leyva",
            "department": "Boyacá",
            "gps_coordinates": {"lat": 5.6359, "lon": -73.5253},
        })
    """

    def __init__(self, orchestrator) -> None:
        self._orchestrator = orchestrator

    async def classify(self, input_data: dict) -> ClassificationResult:
        """
        Clasifica un petroglifo a partir de una imagen y metadatos del sitio.

        Args:
            input_data: Diccionario con:
                - image_path (str): Ruta a la imagen del petroglifo.
                - site (str): Nombre del sitio arqueológico.
                - municipality (str): Municipio.
                - department (str): Departamento.
                - gps_coordinates (dict): {'lat': float, 'lon': float}.
                - conservation_status (str, opcional): Estado de conservación.
                - researcher_notes (str, opcional): Notas del investigador.
                - petroglyph_id (str, opcional): ID existente o se genera uno nuevo.

        Returns:
            ClassificationResult con taxonomy, confidence y justification.

        Raises:
            ValueError: Si image_path no está presente.
        """
        image_path = input_data.get("image_path", "")
        if not image_path:
            raise ValueError("El campo 'image_path' es obligatorio.")

        task_id = input_data.get("petroglyph_id") or str(uuid.uuid4())

        site_metadata = {
            "site": input_data.get("site", "Sin nombre"),
            "municipality": input_data.get("municipality", ""),
            "department": input_data.get("department", ""),
            "gps_coordinates": input_data.get("gps_coordinates", {}),
            "conservation_status": input_data.get("conservation_status", "Regular"),
            "researcher_notes": input_data.get("researcher_notes", ""),
            "site_id": input_data.get("site_id", ""),
        }

        log.info(
            "use_case_classify_start",
            task_id=task_id,
            site=site_metadata["site"],
            municipality=site_metadata["municipality"],
        )

        pipeline_result = await self._orchestrator.run(
            task_id=task_id,
            raw_image_path=image_path,
            site_metadata=site_metadata,
        )

        if pipeline_result.get("status") == "error":
            log.error("use_case_classify_failed", task_id=task_id, error=pipeline_result.get("error"))
            return ClassificationResult(
                taxonomy="Indeterminado",
                confidence=0.0,
                justification=f"Error en pipeline: {pipeline_result.get('error', 'desconocido')}",
                requires_validation=True,
                status="error",
            )

        classification = pipeline_result.get("classification", {})
        return ClassificationResult(
            taxonomy=classification.get("taxonomy", "Indeterminado"),
            confidence=float(classification.get("confidence", 0.0)),
            justification=classification.get("justification", ""),
            retrieved_context=classification.get("retrieved_context", []),
            requires_validation=classification.get("requires_validation", True),
            low_context_quality=classification.get("low_context_quality", False),
            status="success",
        )