"""Evalua las metricas tecnicas del sistema de petroglifos."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from core.metrics import (
    build_metrics_report,
    calculate_autonomous_success_rate,
    calculate_record_sheet_time_metrics,
    detection_metrics_from_grouped_boxes,
    fid_between_image_dirs,
    load_box_samples,
    load_run_records,
)


def build_report(args: argparse.Namespace) -> dict:
    report: dict = {}

    if args.runs_jsonl:
        runs = load_run_records(args.runs_jsonl)
        report.update(
            build_metrics_report(
                runs,
                iou_threshold=args.iou_threshold,
                fid_batch_size=args.fid_batch_size,
                time_threshold_minutes=args.time_threshold_minutes,
            )
        )

    if args.detections_gt and args.detections_pred:
        gt = load_box_samples(args.detections_gt)
        pred = load_box_samples(args.detections_pred)
        report["detection"] = detection_metrics_from_grouped_boxes(
            pred,
            gt,
            iou_threshold=args.iou_threshold,
        )

    if args.real_dir and args.generated_dir:
        report["fid"] = fid_between_image_dirs(
            args.real_dir,
            args.generated_dir,
            batch_size=args.fid_batch_size,
        )

    return report


def print_report(report: dict) -> None:
    if "detection" in report:
        detection = report["detection"]
        print("Detection")
        print(f"  mean_iou: {detection['mean_iou']}")
        print(f"  median_iou: {detection['median_iou']}")
        print(f"  precision_at_threshold: {detection['precision_at_threshold']}")
        print(f"  recall_at_threshold: {detection['recall_at_threshold']}")
        print(f"  success_rate_at_threshold: {detection['success_rate_at_threshold']}")
        print(f"  matched_pairs: {detection['matched_pairs']}")
        print(f"  images_evaluated: {detection['images_evaluated']}")

    if "fid" in report:
        fid = report["fid"]
        print("FID")
        print(f"  fid: {fid['fid']}")
        print(f"  real_images: {fid['real_images']}")
        print(f"  generated_images: {fid['generated_images']}")

    if "autonomous_success" in report:
        success = report["autonomous_success"]
        print("Autonomous Success")
        print(f"  autonomous_success_rate: {success['autonomous_success_rate']}")
        print(f"  successful_runs: {success['successful_runs']}")
        print(f"  total_runs: {success['total_runs']}")

    if "record_sheet_time" in report:
        time_metrics = report["record_sheet_time"]
        print("Record Sheet Time")
        print(f"  mean_minutes: {time_metrics['mean_minutes']}")
        print(f"  median_minutes: {time_metrics['median_minutes']}")
        print(f"  under_threshold_rate: {time_metrics['under_threshold_rate']}")
        print(f"  threshold_minutes: {time_metrics['threshold_minutes']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calcula IoU, FID, tasa de exito autonomo y tiempo de ficha.",
    )
    parser.add_argument("--detections-gt", type=str, help="CSV o JSONL con cajas ground truth")
    parser.add_argument("--detections-pred", type=str, help="CSV o JSONL con cajas predichas")
    parser.add_argument("--real-dir", type=str, help="Directorio de imagenes reales para FID")
    parser.add_argument("--generated-dir", type=str, help="Directorio de imagenes generadas para FID")
    parser.add_argument(
        "--runs-jsonl",
        type=str,
        help="JSONL con ejecuciones del pipeline (storage/metrics/runs.jsonl)",
    )
    parser.add_argument("--output", type=str, help="Ruta para guardar el reporte JSON")
    parser.add_argument("--iou-threshold", type=float, default=0.82)
    parser.add_argument("--fid-batch-size", type=int, default=16)
    parser.add_argument("--time-threshold-minutes", type=float, default=45.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = build_report(args)
    if not report:
        print("No se encontraron insumos para evaluar metricas.")
        return 1

    print_report(report)

    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
