from __future__ import annotations

import numpy as np

from core.metrics import (
    BoxSample,
    calculate_autonomous_success_rate,
    calculate_record_sheet_time_metrics,
    box_iou,
    detection_metrics_from_grouped_boxes,
    fid_from_activations,
)


def test_box_iou_handles_overlap_and_disjoint_boxes() -> None:
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert round(box_iou((0, 0, 10, 10), (5, 5, 15, 15)), 4) == 0.1429


def test_detection_metrics_matches_boxes_and_computes_threshold_success() -> None:
    preds = {
        "img-1": [BoxSample("img-1", (0, 0, 10, 10), 0.9)],
        "img-2": [BoxSample("img-2", (0, 0, 8, 8), 0.8)],
    }
    gts = {
        "img-1": [BoxSample("img-1", (0, 0, 10, 10), 1.0)],
        "img-2": [BoxSample("img-2", (1, 1, 9, 9), 1.0)],
    }

    metrics = detection_metrics_from_grouped_boxes(preds, gts, iou_threshold=0.82)

    assert metrics["matched_pairs"] == 2
    assert metrics["mean_iou"] == 0.8101
    assert metrics["success_rate_at_threshold"] == 0.5


def test_success_and_time_metrics_are_computed_from_run_records() -> None:
    records = [
        {"autonomous_success": True, "total_time_ms": 600_000},
        {"autonomous_success": False, "total_time_ms": 1_200_000},
        {"autonomous_success": True, "total_time_ms": 2_400_000},
    ]

    success = calculate_autonomous_success_rate(records)
    time_metrics = calculate_record_sheet_time_metrics(records, threshold_minutes=45.0)

    assert success["total_runs"] == 3
    assert success["successful_runs"] == 2
    assert success["autonomous_success_rate"] == 0.6667

    assert time_metrics["mean_minutes"] == 23.33
    assert time_metrics["median_minutes"] == 20.0
    assert time_metrics["under_threshold_rate"] == 1.0


def test_fid_is_zero_for_identical_activations() -> None:
    activations = np.array(
        [
            [0.0, 1.0, 2.0],
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 4.0],
        ],
        dtype=float,
    )

    fid = fid_from_activations(activations, activations.copy())

    assert abs(fid) < 1e-6
