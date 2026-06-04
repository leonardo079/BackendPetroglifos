"""A2 — Detector de motivos (YOLOv8 + fallback heurístico con OpenCV)."""
from __future__ import annotations
import base64
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

# Escala humana de conservación para complementar la decisión automática.
_CONSERVATION_SCORE_MAP = {
    "bueno": 0.0,
    "regular": 0.33,
    "malo": 0.75,
    "critico": 1.0,
    "crítico": 1.0,
}


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
        conservation_status = self._normalize_conservation_status(
            input.payload.get("conservation_status", "Regular")
        )
        conservation_score = self._conservation_score(conservation_status)
        human_reconstruction_recommended = conservation_score >= 0.75
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
        detection.update(
            {
                "conservation_status": conservation_status,
                "conservation_score": conservation_score,
                "human_reconstruction_recommended": human_reconstruction_recommended,
                "reconstruction_recommended": (
                    detection.get("deterioration_detected", False)
                    or human_reconstruction_recommended
                ),
                "reconstruction_assessment": {
                    "model_deterioration_detected": detection.get("deterioration_detected", False),
                    "model_damage_recommended": detection.get("model_damage_recommended", False),
                    "damage_figure_percent": detection.get("damage_figure_percent"),
                    "conservation_status": conservation_status,
                    "conservation_score": conservation_score,
                    "human_reconstruction_recommended": human_reconstruction_recommended,
                    "reconstruction_recommended": (
                        detection.get("deterioration_detected", False)
                        or human_reconstruction_recommended
                    ),
                },
            }
        )

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("a2_detector_done",
                 task_id=input.task_id,
                 motifs_visible=detection["motifs_visible"],
                 shapes=detection["detected_shapes"],
                 deterioration=detection["deterioration_detected"],
                 conservation_status=conservation_status,
                 conservation_score=conservation_score,
                 human_reconstruction_recommended=human_reconstruction_recommended,
                 segmentation_score=detection.get("segmentation_score"),
                 damage_figure_percent=detection.get("damage_figure_percent"),
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
        Determina si la imagen tiene deterioro significativo combinando dos señales
        del servicio Keras:

        1. Calidad de la segmentación de la figura (/segmentPetroglyph). Cualquiera de:
          - validation_score < 0        : máscara de baja calidad
          - segmentation_status == 'weak_segmentation'
          - 'fragmented_mask' en warnings : surcos muy fragmentados
          - 'weak_main_component' en warnings : componente principal débil
          - area_percent < 6.0          : cobertura visible insuficiente

        2. Daño de la figura (/segmentDamagePytorch). Se cruza la máscara de daño con
           la máscara de la figura y se calcula la fracción de la figura dañada. Si
           supera settings.damage_reconstruction_threshold se fuerza reconstrucción.

        La decisión final es el OR de ambas señales.

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

            quality_deterioration = (
                score < 0
                or status == "weak_segmentation"
                or "fragmented_mask" in warnings
                or "weak_main_component" in warnings
                or area_percent < 6.0
            )

            # Señal de daño: cruzar la máscara de la figura con la máscara de daño.
            figure_mask = self._decode_mask_b64(data.get("mask_image"))
            damage_ratio = await self._figure_damage_ratio_from_api(image_path, figure_mask)
            damage_figure_percent = (
                round(damage_ratio * 100, 2) if damage_ratio is not None else None
            )
            damage_deterioration = (
                damage_ratio is not None
                and damage_ratio >= settings.damage_reconstruction_threshold
            )

            deterioration_detected = quality_deterioration or damage_deterioration

            log.info(
                "a2_deterioration_api",
                score=score,
                status=status,
                warnings=warnings,
                warnings_count=len(warnings),
                area_percent=area_percent,
                damage_figure_percent=damage_figure_percent,
                damage_deterioration=damage_deterioration,
                deterioration=deterioration_detected,
            )
            return {
                "deterioration_detected": deterioration_detected,
                "segmentation_score": score,
                "segmentation_status": status,
                "segmentation_warnings": warnings,
                "area_percent": area_percent,
                "damage_figure_percent": damage_figure_percent,
                "model_damage_recommended": damage_deterioration,
                "segmentation_validation": {
                    "validation_score": score,
                    "segmentation_status": status,
                    "validation_warnings": warnings,
                    "area_percent": area_percent,
                    "damage_figure_percent": damage_figure_percent,
                    "model_damage_recommended": damage_deterioration,
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
                "damage_figure_percent": None,
                "model_damage_recommended": False,
                "segmentation_validation": {
                    "validation_score": 0.0,
                    "segmentation_status": "unknown",
                    "validation_warnings": ["api_unavailable"],
                    "area_percent": 0.0,
                    "damage_figure_percent": None,
                    "model_damage_recommended": False,
                    "deterioration_detected": True,
                },
            }

    async def _figure_damage_ratio_from_api(
        self, image_path: str, figure_mask: np.ndarray | None
    ) -> float | None:
        """
        Consulta /segmentDamagePytorch y devuelve la fracción (0.0-1.0) de la figura
        que está dañada (daño ∩ figura / área de la figura).

        Retorna None si no se puede determinar (modelo de daño no disponible,
        figura no detectada, etc.), en cuyo caso la decisión recae solo en los
        criterios de calidad.
        """
        if figure_mask is None:
            return None
        damage_url = f"{settings.reconstruction_api_base_url}/segmentDamagePytorch"
        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                with open(image_path, "rb") as f:
                    response = await client.post(
                        damage_url,
                        data={"save_png": "false"},
                        files={"file": (Path(image_path).name, f, "image/jpeg")},
                    )
                response.raise_for_status()
                damage_data = response.json()
            damage_mask = self._decode_mask_b64(damage_data.get("mask_image"))
            return self._figure_damage_ratio(figure_mask, damage_mask)
        except Exception as exc:
            log.warning("a2_damage_api_failed", error=str(exc), fallback="quality_only")
            return None

    @staticmethod
    def _decode_mask_b64(b64_str: object) -> np.ndarray | None:
        """Decodifica una máscara base64 (PNG en escala de grises) a un array 2D."""
        if not b64_str or not isinstance(b64_str, str):
            return None
        try:
            raw = base64.b64decode(b64_str)
            buf = np.frombuffer(raw, np.uint8)
            return cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        except Exception:
            return None

    @staticmethod
    def _figure_damage_ratio(
        petro_mask: np.ndarray | None, damage_mask: np.ndarray | None
    ) -> float | None:
        """daño ∩ figura / área_figura, como fracción 0.0-1.0."""
        if petro_mask is None or damage_mask is None:
            return None
        if damage_mask.shape != petro_mask.shape:
            damage_mask = cv2.resize(
                damage_mask,
                (petro_mask.shape[1], petro_mask.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            )
        figure = petro_mask > 127
        figure_area = int(figure.sum())
        if figure_area == 0:
            return None
        damaged = figure & (damage_mask > 127)
        return float(int(damaged.sum()) / figure_area)

    @staticmethod
    def _normalize_conservation_status(value: object) -> str:
        if value is None:
            return "Regular"
        text = str(value).strip()
        if not text:
            return "Regular"
        lowered = text.lower()
        if lowered in _CONSERVATION_SCORE_MAP:
            return "Crítico" if lowered == "crítico" else text.capitalize()
        if lowered == "critico":
            return "Crítico"
        if lowered == "bueno":
            return "Bueno"
        if lowered == "regular":
            return "Regular"
        if lowered == "malo":
            return "Malo"
        return text

    @staticmethod
    def _conservation_score(status: str) -> float:
        return float(_CONSERVATION_SCORE_MAP.get(status.lower(), 0.33))

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
