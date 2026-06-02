"""A2 — Detector de motivos (YOLOv8 + fallback heurístico con OpenCV)."""
from __future__ import annotations
import time
import os
from pathlib import Path
import cv2
import numpy as np
import httpx
import structlog
from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from config.settings import settings

log = structlog.get_logger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "petroglifos_yolov8.pt"

# Categorías de formas detectables con el método heurístico
SHAPE_LABELS = {
    3: "Triángulo",
    4: "Cuadrilátero",
    5: "Pentágono",
    6: "Hexágono",
}

# Mapeo de nombres del modelo (minúsculas, sin tildes) al vocabulario
# controlado de TaxonomyCategory (mayúsculas, con tildes).
_TAXONOMY_MAP = {
    "antropomorfo": "Antropomorfo",
    "zoomorfo":     "Zoomorfo",
    "geometrico":   "Geométrico",
    "fitomorfo":    "Fitomorfo",
    "astronomico":  "Astronómico",
    "hibrido":      "Híbrido",
}

# Confianza mínima para incluir clases alternativas (top-2, top-3) en detected_shapes.
_ALT_CLASS_THRESHOLD = 0.20


class DetectorAgent(BaseAgent):
    name = "a2_detector"

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._yolo = None
        self._load_yolo()

    def _load_yolo(self) -> None:
        if YOLO_MODEL_PATH.exists():
            try:
                from ultralytics import YOLO
                self._yolo = YOLO(str(YOLO_MODEL_PATH))
                log.info(
                    "yolo_loaded",
                    model=str(YOLO_MODEL_PATH),
                    cwd=str(Path.cwd()),
                )
            except Exception as e:
                log.warning(
                    "yolo_load_failed",
                    error=str(e),
                    model=str(YOLO_MODEL_PATH),
                    cwd=str(Path.cwd()),
                    fallback="heuristic",
                )
        else:
            log.warning(
                "yolo_model_not_found",
                path=str(YOLO_MODEL_PATH),
                cwd=str(Path.cwd()),
                fallback="heuristic",
            )

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("preprocessed_image_path", "") or \
                          input.payload.get("image_path", "")
        log.info(
            "a2_detector_input",
            task_id=input.task_id,
            image_path=image_path,
            yolo_loaded=self._yolo is not None,
            cwd=str(Path.cwd()),
        )

        if not image_path or not os.path.exists(image_path):
            return AgentOutput(
                task_id=input.task_id, agent_name=self.name,
                result={}, status="error",
                metadata={"error": f"Imagen no encontrada: {image_path}"},
            )

        img = cv2.imread(image_path)
        if img is None:
            return AgentOutput(task_id=input.task_id, agent_name=self.name, result={}, status="error")

        # 1. Detección de motivos (síncrona)
        if self._yolo is not None:
            if self._is_cls_model():
                detection = self._detect_yolo_cls(img, image_path)
            else:
                detection = self._detect_yolo(img, image_path)
        else:
            detection = self._detect_heuristic(img)

        # 2. Deterioro via API Keras /segmentPetroglyph (reemplaza Laplaciano)
        deterioration = await self._check_deterioration_api(image_path)
        detection.update(deterioration)

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("a2_detector_done",
                 task_id=input.task_id,
                 motifs_visible=detection["motifs_visible"],
                 shapes=detection["detected_shapes"],
                 deterioration=detection["deterioration_detected"],
                 segmentation_score=detection.get("segmentation_score"),
                 latency_ms=elapsed)

        return AgentOutput(
            task_id=input.task_id,
            agent_name=self.name,
            result=detection,
            status="success",
            metadata={"latency_ms": elapsed, "method": detection.get("method", "unknown")},
        )

    def _is_cls_model(self) -> bool:
        """Determina si el modelo cargado es de clasificación (no detección)."""
        # Método 1: atributo task (disponible en Ultralytics >= 8.0)
        if getattr(self._yolo, "task", None) == "classify":
            return True
        # Método 2: inspección de la última capa del modelo
        model_obj = getattr(self._yolo, "model", None)
        if model_obj is not None:
            head = getattr(model_obj, "model", None)
            if head is not None:
                children = list(head.children())
                if children and "Classify" in type(children[-1]).__name__:
                    return True
        return False

    def _detect_yolo_cls(self, img: np.ndarray, image_path: str) -> dict:
        """Procesa la salida de un modelo YOLOv8-cls (r.probs)."""
        h, w = img.shape[:2]
        results = self._yolo(image_path, verbose=False)

        shapes: list[str] = []
        boxes: list[dict] = []
        top_confidence = 0.0

        for r in results or []:
            probs = getattr(r, "probs", None)
            if probs is None:
                continue

            names = self._yolo.names
            top1_idx = int(probs.top1)
            top1_conf = float(probs.top1conf)
            top1_label = self._canonicalize(names[top1_idx])

            # Clase principal: solo se reporta como motivo visible si supera
            # el umbral. Si no, la imagen irá a A5 y no añadimos alternativas
            # de baja confianza que contradigan motifs_visible=False.
            if top1_conf < settings.confidence_threshold:
                continue

            shapes.append(top1_label)
            boxes.append({
                "label": top1_label,
                "confidence": round(top1_conf, 4),
                "xyxy": [0, 0, w, h],
            })
            top_confidence = max(top_confidence, top1_conf)

            # Clases alternativas (top-2, top-3...) con confianza > 0.20,
            # para dar más contexto a A4.
            for idx, conf in zip(list(probs.top5), [float(c) for c in probs.top5conf]):
                if idx == top1_idx or conf < _ALT_CLASS_THRESHOLD:
                    continue
                label = self._canonicalize(names[idx])
                if label not in shapes:
                    shapes.append(label)

        log.info(
            "a2_yolo_cls_result",
            image_path=image_path,
            classes=shapes,
            confidence=top_confidence,
        )

        return {
            "motifs_visible": len(boxes) > 0,
            "detected_shapes": list(dict.fromkeys(shapes)),
            "bounding_boxes": boxes,
            "detection_confidence": top_confidence,
            "motif_description": self._describe(shapes),
            "method": "yolov8_cls",
        }

    @staticmethod
    def _canonicalize(label: str) -> str:
        """Mapea el nombre del modelo al vocabulario controlado de TaxonomyCategory."""
        return _TAXONOMY_MAP.get(label.strip().lower(), label)

    def _detect_yolo(self, img: np.ndarray, image_path: str) -> dict:
        results = self._yolo(image_path, conf=settings.confidence_threshold, verbose=False)
        boxes = []
        shapes: list[str] = []
        result_count = 0
        raw_box_count = 0
        for r in results or []:
            result_count += 1
            result_boxes = getattr(r, "boxes", None) or []
            raw_box_count += len(result_boxes)
            for box in result_boxes:
                cls_name = self._yolo.names[int(box.cls)]
                shapes.append(cls_name)
                boxes.append({
                    "label": cls_name,
                    "confidence": float(box.conf),
                    "xyxy": box.xyxy[0].tolist(),
                })

        log.info(
            "a2_yolo_result",
            image_path=image_path,
            results_count=result_count,
            raw_box_count=raw_box_count,
            filtered_box_count=len(boxes),
            classes=shapes,
            confidence=max((b["confidence"] for b in boxes), default=0.0),
        )

        return {
            "motifs_visible": len(boxes) > 0,
            "detected_shapes": list(set(shapes)),
            "bounding_boxes": boxes,
            "detection_confidence": max((b["confidence"] for b in boxes), default=0.0),
            "motif_description": self._describe(shapes),
            "method": "yolov8",
        }

    def _detect_heuristic(self, img: np.ndarray) -> dict:
        """Detección heurística con OpenCV cuando YOLOv8 no está disponible."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 2)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes: list[dict] = []
        shapes: list[str] = []
        min_area = img.shape[0] * img.shape[1] * 0.005  # >0.5% de la imagen

        for cnt in contours:
            if cv2.contourArea(cnt) < min_area:
                continue
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            n = len(approx)
            shape = SHAPE_LABELS.get(n, "Círculo" if n > 6 else "Forma irregular")
            x, y, w, h = cv2.boundingRect(cnt)
            shapes.append(shape)
            boxes.append({
                "label": shape,
                "confidence": 0.5,
                "xyxy": [x, y, x + w, y + h],
            })

        # Limitar a los 10 contornos más grandes
        boxes = sorted(boxes, key=lambda b: (b["xyxy"][2]-b["xyxy"][0])*(b["xyxy"][3]-b["xyxy"][1]), reverse=True)[:10]
        shapes = [b["label"] for b in boxes]

        log.info(
            "a2_heuristic_result",
            boxes_count=len(boxes),
            classes=shapes,
            confidence=0.5 if boxes else 0.0,
        )

        return {
            "motifs_visible": len(boxes) > 0,
            "detected_shapes": list(set(shapes)),
            "bounding_boxes": boxes,
            "detection_confidence": 0.5 if boxes else 0.0,
            "motif_description": self._describe(shapes),
            "method": "heuristic_opencv",
        }

    def _describe(self, shapes: list[str]) -> str:
        if not shapes:
            return "No se detectaron motivos claros en la imagen."
        counts: dict[str, int] = {}
        for s in shapes:
            counts[s] = counts.get(s, 0) + 1
        parts = [f"{v} {k}{'s' if v > 1 else ''}" for k, v in counts.items()]
        return f"Se detectaron: {', '.join(parts)}."

    async def _check_deterioration_api(self, image_path: str) -> dict:
        """
        Consulta /segmentPetroglyph del servicio Keras y determina si la imagen
        tiene deterioro significativo a partir del validation_score y los warnings
        producidos por el modelo de segmentación.

        Criterios de deterioro (cualquiera de los siguientes):
          - validation_score < 0        : máscara de baja calidad
          - segmentation_status == 'weak_segmentation'
          - 'fragmented_mask' en warnings : surcos muy fragmentados
          - 'weak_main_component' en warnings : componente principal débil
          - area_percent < 2.0          : prácticamente sin petroglifo visible

        Fallback (API no disponible): asume deterioro para forzar reconstrucción
        y no perderse figuras dañadas.
        """
        segment_url = f"{settings.reconstruction_api_base_url}/segmentPetroglyph"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(image_path, "rb") as f:
                    response = await client.post(
                        segment_url,
                        data={"include_previews": "false"},
                        files={"file": (Path(image_path).name, f, "image/jpeg")},
                    )
                response.raise_for_status()
                data = response.json()

            score = float(data.get("validation_score", 0.0))
            status = str(data.get("segmentation_status", "ok"))
            warnings = self._normalize_warnings(data.get("validation_warnings"))
            area_percent = float(data.get("area_percent", 0.0))

            deterioration_detected = (
                score < 0
                or status == "weak_segmentation"
                or "fragmented_mask" in warnings
                or "weak_main_component" in warnings
                or area_percent < 2.0
            )

            log.info(
                "a2_deterioration_api",
                score=score,
                status=status,
                warnings=warnings,
                warnings_count=len(warnings),
                area_percent=area_percent,
                deterioration=deterioration_detected,
            )
            return {
                "deterioration_detected": deterioration_detected,
                "segmentation_score": score,
                "segmentation_status": status,
                "segmentation_warnings": warnings,
                "area_percent": area_percent,
                "segmentation_validation": {
                    "validation_score": score,
                    "segmentation_status": status,
                    "validation_warnings": warnings,
                    "area_percent": area_percent,
                    "deterioration_detected": deterioration_detected,
                },
            }

        except Exception as e:
            log.warning(
                "a2_deterioration_api_failed",
                error=str(e),
                fallback="assume_deteriorated",
            )
            # Conservador: si la API falla, asumir deterioro para no omitir
            # figuras que necesiten reconstrucción.
            return {
                "deterioration_detected": True,
                "segmentation_score": 0.0,
                "segmentation_status": "unknown",
                "segmentation_warnings": ["api_unavailable"],
                "area_percent": 0.0,
                "segmentation_validation": {
                    "validation_score": 0.0,
                    "segmentation_status": "unknown",
                    "validation_warnings": ["api_unavailable"],
                    "area_percent": 0.0,
                    "deterioration_detected": True,
                },
            }

    def _normalize_warnings(self, warnings: object) -> list[str]:
        if warnings is None:
            return []
        if isinstance(warnings, list):
            return [str(item) for item in warnings if item is not None and str(item).strip()]
        if isinstance(warnings, tuple) or isinstance(warnings, set):
            return [str(item) for item in warnings if item is not None and str(item).strip()]
        if isinstance(warnings, str):
            text = warnings.strip()
            return [text] if text else []
        return [str(warnings)] if str(warnings).strip() else []
