"""Utilidades de metrica para evaluacion tecnica del pipeline."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
import csv
import json

import numpy as np

try:  # scipy mejora la estabilidad del calculo de FID.
    from scipy import linalg
except Exception:  # pragma: no cover - fallback defensivo
    linalg = None


Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class BoxSample:
    image_id: str
    box: Box
    score: float = 1.0


def box_iou(box_a: Box, box_b: Box) -> float:
    """Calcula IoU entre dos cajas en formato (x1, y1, x2, y2)."""
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area

    if union <= 0.0:
        return 0.0
    return float(inter_area / union)


def greedy_match_ious(
    predictions: Sequence[BoxSample],
    ground_truth: Sequence[BoxSample],
) -> list[float]:
    """Empareja predicciones con GT de mayor score a menor score."""
    matched_ious: list[float] = []
    remaining_gt = list(range(len(ground_truth)))
    ordered_predictions = sorted(predictions, key=lambda sample: sample.score, reverse=True)

    for pred in ordered_predictions:
        best_iou = 0.0
        best_idx: int | None = None
        for idx in remaining_gt:
            iou = box_iou(pred.box, ground_truth[idx].box)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx is not None:
            matched_ious.append(best_iou)
            remaining_gt.remove(best_idx)

    return matched_ious


def detection_metrics_from_grouped_boxes(
    predicted_by_image: dict[str, Sequence[BoxSample]],
    ground_truth_by_image: dict[str, Sequence[BoxSample]],
    iou_threshold: float = 0.82,
) -> dict:
    """Calcula metrics de deteccion sobre un conjunto de imagenes."""
    all_ious: list[float] = []
    total_predictions = 0
    total_ground_truth = 0

    image_ids = sorted(set(predicted_by_image) | set(ground_truth_by_image))
    for image_id in image_ids:
        preds = list(predicted_by_image.get(image_id, []))
        gts = list(ground_truth_by_image.get(image_id, []))
        total_predictions += len(preds)
        total_ground_truth += len(gts)
        all_ious.extend(greedy_match_ious(preds, gts))

    matched_count = len(all_ious)
    successful_matches = sum(iou >= iou_threshold for iou in all_ious)
    mean_iou = float(np.mean(all_ious)) if all_ious else 0.0
    median_iou = float(np.median(all_ious)) if all_ious else 0.0
    precision_at_threshold = float(successful_matches / total_predictions) if total_predictions else 0.0
    recall_at_threshold = float(successful_matches / total_ground_truth) if total_ground_truth else 0.0
    return {
        "images_evaluated": len(image_ids),
        "predictions": total_predictions,
        "ground_truth": total_ground_truth,
        "matched_pairs": matched_count,
        "successful_matches": successful_matches,
        "mean_iou": round(mean_iou, 4),
        "median_iou": round(median_iou, 4),
        "precision_at_threshold": round(precision_at_threshold, 4),
        "recall_at_threshold": round(recall_at_threshold, 4),
        "success_rate_at_threshold": round(recall_at_threshold, 4),
        "iou_threshold": iou_threshold,
    }


def load_box_samples(path: str | Path) -> dict[str, list[BoxSample]]:
    """
    Carga cajas desde CSV o JSONL.

    Formato CSV esperado:
        image_id,x1,y1,x2,y2,score

    Formato JSONL esperado:
        {"image_id":"...", "x1":0, "y1":0, "x2":10, "y2":10, "score":0.93}
    """
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No existe el archivo de cajas: {source}")

    if source.suffix.lower() == ".csv":
        rows = _load_box_csv(source)
    elif source.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = _load_box_jsonl(source)
    else:
        raise ValueError("Use un archivo .csv, .jsonl o .ndjson para las cajas")

    grouped: dict[str, list[BoxSample]] = defaultdict(list)
    for row in rows:
        grouped[row.image_id].append(row)
    return dict(grouped)


def _load_box_csv(path: Path) -> list[BoxSample]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"image_id", "x1", "y1", "x2", "y2"}
        if not required.issubset(reader.fieldnames or set()):
            missing = ", ".join(sorted(required - set(reader.fieldnames or [])))
            raise ValueError(f"Faltan columnas requeridas en {path}: {missing}")

        samples: list[BoxSample] = []
        for row in reader:
            samples.append(_row_to_sample(row))
    return samples


def _load_box_jsonl(path: Path) -> list[BoxSample]:
    samples: list[BoxSample] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        row = json.loads(raw_line)
        samples.append(_row_to_sample(row))
    return samples


def _row_to_sample(row: dict) -> BoxSample:
    image_id = str(row["image_id"]).strip()
    box = (
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    )
    score = float(row.get("score", 1.0) or 1.0)
    return BoxSample(image_id=image_id, box=box, score=score)


def _dict_to_box_sample(image_id: str, row: dict) -> BoxSample:
    box = (
        float(row["x1"]),
        float(row["y1"]),
        float(row["x2"]),
        float(row["y2"]),
    )
    score = float(row.get("score", 1.0) or 1.0)
    return BoxSample(image_id=image_id, box=box, score=score)


def calculate_autonomous_success_rate(records: Iterable[dict]) -> dict:
    """Calcula tasa de exito autonomo a partir de registros del pipeline."""
    items = list(records)
    total = len(items)
    if total == 0:
        return {
            "total_runs": 0,
            "successful_runs": 0,
            "autonomous_success_rate": 0.0,
        }

    successful = sum(1 for item in items if bool(item.get("autonomous_success", False)))
    return {
        "total_runs": total,
        "successful_runs": successful,
        "autonomous_success_rate": round(successful / total, 4),
    }


def calculate_record_sheet_time_metrics(records: Iterable[dict], threshold_minutes: float = 45.0) -> dict:
    """Calcula tiempo medio y cumplimiento del umbral por ficha."""
    items = list(records)
    times_ms = [float(item["total_time_ms"]) for item in items if item.get("total_time_ms") is not None]
    if not times_ms:
        return {
            "total_runs": len(items),
            "mean_minutes": 0.0,
            "median_minutes": 0.0,
            "under_threshold_rate": 0.0,
            "threshold_minutes": threshold_minutes,
        }

    times_minutes = [ms / 60000.0 for ms in times_ms]
    under_threshold = sum(minutes < threshold_minutes for minutes in times_minutes)
    return {
        "total_runs": len(times_minutes),
        "mean_minutes": round(float(np.mean(times_minutes)), 2),
        "median_minutes": round(float(np.median(times_minutes)), 2),
        "under_threshold_rate": round(under_threshold / len(times_minutes), 4),
        "threshold_minutes": threshold_minutes,
    }


def load_run_records(path: str | Path) -> list[dict]:
    """Carga registros de ejecucion desde un JSONL."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"No existe el archivo de ejecuciones: {source}")

    records: list[dict] = []
    for raw_line in source.read_text(encoding="utf-8").splitlines():
        if not raw_line.strip():
            continue
        records.append(json.loads(raw_line))
    return records


def detection_metrics_from_run_records(records: Iterable[dict], iou_threshold: float = 0.82) -> dict | None:
    """
    Calcula metrics de deteccion a partir de corridas acumuladas.

    Se espera que cada registro pueda contener:
    - `detection_pred_boxes`
    - `detection_gt_boxes`
    - `image_id` o `task_id`
    """
    predicted_by_image: dict[str, list[BoxSample]] = defaultdict(list)
    ground_truth_by_image: dict[str, list[BoxSample]] = defaultdict(list)

    for record in records:
        image_id = str(record.get("image_id") or record.get("task_id") or "").strip()
        if not image_id:
            continue

        for pred in record.get("detection_pred_boxes", []) or []:
            predicted_by_image[image_id].append(_dict_to_box_sample(image_id, pred))

        for gt in record.get("detection_gt_boxes", []) or []:
            ground_truth_by_image[image_id].append(_dict_to_box_sample(image_id, gt))

    if not predicted_by_image or not ground_truth_by_image:
        return None

    return detection_metrics_from_grouped_boxes(
        dict(predicted_by_image),
        dict(ground_truth_by_image),
        iou_threshold=iou_threshold,
    )


def image_pairs_from_runs(records: Iterable[dict]) -> tuple[list[Path], list[Path]]:
    """
    Extrae listas de imagenes reales y generadas desde las corridas.

    Usa raw_image_path como referencia real y reconstructed_image_path como
    imagen generada; si no existe reconstruccion, cae a preprocessed_image_path.
    """
    real_paths: list[Path] = []
    generated_paths: list[Path] = []

    for record in records:
        raw_path = record.get("raw_image_path")
        generated_path = record.get("reconstructed_image_path") or record.get("preprocessed_image_path")
        if not raw_path or not generated_path:
            continue

        real = Path(raw_path)
        generated = Path(generated_path)
        if real.exists() and generated.exists():
            real_paths.append(real)
            generated_paths.append(generated)

    return real_paths, generated_paths


def image_files(root: str | Path) -> list[Path]:
    """Lista imagenes soportadas dentro de un directorio."""
    root_path = Path(root)
    if not root_path.exists():
        raise FileNotFoundError(f"No existe el directorio de imagenes: {root_path}")

    suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
    return sorted(path for path in root_path.rglob("*") if path.suffix.lower() in suffixes)


def extract_image_activations(image_paths: Sequence[Path], batch_size: int = 16) -> np.ndarray:
    """
    Extrae activaciones de InceptionV3 para FID.

    Requiere torchvision y, si el modelo no esta cacheado, puede descargar
    pesos preentrenados.
    """
    if not image_paths:
        raise ValueError("Se requiere al menos una imagen para calcular FID")

    import torch
    from PIL import Image
    from torchvision.models import Inception_V3_Weights, inception_v3

    weights = Inception_V3_Weights.DEFAULT
    preprocess = weights.transforms()

    model = inception_v3(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    activations: list[np.ndarray] = []
    batch: list[torch.Tensor] = []

    with torch.no_grad():
        for path in image_paths:
            image = Image.open(path).convert("RGB")
            tensor = preprocess(image)
            batch.append(tensor)
            if len(batch) < batch_size:
                continue

            activations.append(_forward_batch(model, batch, device))
            batch = []

        if batch:
            activations.append(_forward_batch(model, batch, device))

    return np.concatenate(activations, axis=0)


def _forward_batch(model, batch: list, device) -> np.ndarray:
    import torch

    tensor = torch.stack(batch).to(device)
    features = model(tensor)
    return features.detach().cpu().numpy()


def fid_from_activations(real_activations: np.ndarray, generated_activations: np.ndarray) -> float:
    """Calcula FID a partir de activaciones ya extraidas."""
    if real_activations.ndim != 2 or generated_activations.ndim != 2:
        raise ValueError("Las activaciones deben ser matrices 2D")
    if real_activations.shape[0] < 2 or generated_activations.shape[0] < 2:
        raise ValueError("FID requiere al menos dos imagenes por conjunto")

    mu1 = np.mean(real_activations, axis=0)
    mu2 = np.mean(generated_activations, axis=0)
    sigma1 = np.cov(real_activations, rowvar=False)
    sigma2 = np.cov(generated_activations, rowvar=False)

    return float(_frechet_distance(mu1, sigma1, mu2, sigma2))


def fid_between_image_dirs(real_dir: str | Path, generated_dir: str | Path, batch_size: int = 16) -> dict:
    """Calcula FID entre dos carpetas de imagenes."""
    real_images = image_files(real_dir)
    generated_images = image_files(generated_dir)
    if len(real_images) < 2:
        raise ValueError(f"Se requieren al menos 2 imagenes reales en {real_dir}")
    if len(generated_images) < 2:
        raise ValueError(f"Se requieren al menos 2 imagenes generadas en {generated_dir}")

    real_act = extract_image_activations(real_images, batch_size=batch_size)
    gen_act = extract_image_activations(generated_images, batch_size=batch_size)
    return {
        "real_images": len(real_images),
        "generated_images": len(generated_images),
        "fid": round(fid_from_activations(real_act, gen_act), 4),
    }


def fid_between_image_lists(real_images: Sequence[Path], generated_images: Sequence[Path], batch_size: int = 16) -> dict:
    """Calcula FID entre listas de imagenes ya seleccionadas."""
    if len(real_images) < 2:
        raise ValueError("Se requieren al menos 2 imagenes reales para FID")
    if len(generated_images) < 2:
        raise ValueError("Se requieren al menos 2 imagenes generadas para FID")

    real_act = extract_image_activations(list(real_images), batch_size=batch_size)
    gen_act = extract_image_activations(list(generated_images), batch_size=batch_size)
    return {
        "real_images": len(real_images),
        "generated_images": len(generated_images),
        "fid": round(fid_from_activations(real_act, gen_act), 4),
    }


def build_metrics_report(
    records: Sequence[dict],
    *,
    iou_threshold: float = 0.82,
    fid_batch_size: int = 16,
    time_threshold_minutes: float = 45.0,
) -> dict:
    """Construye un reporte consolidado desde registros acumulados."""
    report: dict = {
        "autonomous_success": calculate_autonomous_success_rate(records),
        "record_sheet_time": calculate_record_sheet_time_metrics(
            records,
            threshold_minutes=time_threshold_minutes,
        ),
    }

    detection = detection_metrics_from_run_records(records, iou_threshold=iou_threshold)
    if detection is not None:
        report["detection"] = detection
    else:
        report["detection"] = {
            "status": "skipped",
            "reason": "no_ground_truth_boxes_available",
        }

    real_images, generated_images = image_pairs_from_runs(records)
    if len(real_images) >= 2 and len(generated_images) >= 2:
        try:
            report["fid"] = fid_between_image_lists(
                real_images,
                generated_images,
                batch_size=fid_batch_size,
            )
        except Exception as exc:
            report["fid"] = {"status": "error", "error": str(exc)}
    else:
        report["fid"] = {
            "status": "skipped",
            "reason": "insufficient_image_pairs",
            "real_images": len(real_images),
            "generated_images": len(generated_images),
        }

    return report


def _frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> float:
    eps = 1e-6
    sigma1 = np.asarray(sigma1) + np.eye(sigma1.shape[0]) * eps
    sigma2 = np.asarray(sigma2) + np.eye(sigma2.shape[0]) * eps
    diff = mu1 - mu2

    if linalg is None:
        raise RuntimeError("scipy no esta disponible; no se puede calcular FID de forma estable")

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return float(np.real(fid))
