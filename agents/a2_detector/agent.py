"""A2 — Detector de motivos (YOLOv8 + fallback heurístico con OpenCV)."""
from __future__ import annotations
import time
import os
from pathlib import Path
import cv2
import numpy as np
import structlog
from agents.base_agent import BaseAgent, AgentInput, AgentOutput
from config.settings import settings

log = structlog.get_logger(__name__)

YOLO_MODEL_PATH = Path("models/petroglifos_yolov8.pt")

# Categorías de formas detectables con el método heurístico
SHAPE_LABELS = {
    3: "Triángulo",
    4: "Cuadrilátero",
    5: "Pentágono",
    6: "Hexágono",
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
                log.info("yolo_loaded", model=str(YOLO_MODEL_PATH))
            except Exception as e:
                log.warning("yolo_load_failed", error=str(e), fallback="heuristic")
        else:
            log.warning("yolo_model_not_found", path=str(YOLO_MODEL_PATH), fallback="heuristic")

    async def run(self, input: AgentInput) -> AgentOutput:
        t0 = time.monotonic()
        image_path: str = input.payload.get("preprocessed_image_path", "") or \
                          input.payload.get("image_path", "")

        if not image_path or not os.path.exists(image_path):
            return AgentOutput(
                task_id=input.task_id, agent_name=self.name,
                result={}, status="error",
                metadata={"error": f"Imagen no encontrada: {image_path}"},
            )

        img = cv2.imread(image_path)
        if img is None:
            return AgentOutput(task_id=input.task_id, agent_name=self.name, result={}, status="error")

        if self._yolo is not None:
            detection = self._detect_yolo(img, image_path)
        else:
            detection = self._detect_heuristic(img)

        elapsed = round((time.monotonic() - t0) * 1000)
        log.info("a2_detector_done",
                 task_id=input.task_id,
                 motifs_visible=detection["motifs_visible"],
                 shapes=detection["detected_shapes"],
                 latency_ms=elapsed)

        return AgentOutput(
            task_id=input.task_id,
            agent_name=self.name,
            result=detection,
            status="success",
            metadata={"latency_ms": elapsed, "method": detection.get("method", "unknown")},
        )

    def _detect_yolo(self, img: np.ndarray, image_path: str) -> dict:
        results = self._yolo(image_path, conf=settings.confidence_threshold, verbose=False)
        boxes = []
        shapes: list[str] = []
        for r in results:
            for box in r.boxes:
                cls_name = self._yolo.names[int(box.cls)]
                shapes.append(cls_name)
                boxes.append({
                    "label": cls_name,
                    "confidence": float(box.conf),
                    "xyxy": box.xyxy[0].tolist(),
                })

        return {
            "motifs_visible": len(boxes) > 0,
            "detected_shapes": list(set(shapes)),
            "bounding_boxes": boxes,
            "detection_confidence": max((b["confidence"] for b in boxes), default=0.0),
            "motif_description": self._describe(shapes),
            "deterioration_detected": self._check_deterioration(img),
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

        return {
            "motifs_visible": len(boxes) > 0,
            "detected_shapes": list(set(shapes)),
            "bounding_boxes": boxes,
            "detection_confidence": 0.5 if boxes else 0.0,
            "motif_description": self._describe(shapes),
            "deterioration_detected": self._check_deterioration(img),
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

    def _check_deterioration(self, img: np.ndarray) -> bool:
        """Heurística simple: alta entropía de bordes → posible deterioro."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var < 50  # baja nitidez → posible deterioro