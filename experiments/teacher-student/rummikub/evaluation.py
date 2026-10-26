"""Small labeled-dataset evaluator for the local Roboflow model."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import cv2

from rummikub_inference import build_client, infer

from .classes import CLASS_NAMES, canonicalize_external_name
from .dataset import IMAGE_DIR, LABEL_DIR, image_choices, load_boxes


def prediction_boxes(result: dict[str, Any]) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    for prediction in result.get("predictions", []):
        try:
            class_name = canonicalize_external_name(str(prediction["class"]))
        except ValueError:
            continue
        width, height = float(prediction["width"]), float(prediction["height"])
        cx, cy = float(prediction["x"]), float(prediction["y"])
        boxes.append({
            "class_id": CLASS_NAMES.index(class_name),
            "class_name": class_name,
            "confidence": float(prediction.get("confidence", 0)),
            "x1": cx - width / 2,
            "y1": cy - height / 2,
            "x2": cx + width / 2,
            "y2": cy + height / 2,
        })
    return boxes


def iou(first: dict[str, Any], second: dict[str, Any]) -> float:
    left, top = max(first["x1"], second["x1"]), max(first["y1"], second["y1"])
    right, bottom = min(first["x2"], second["x2"]), min(first["y2"], second["y2"])
    intersection = max(0.0, right - left) * max(0.0, bottom - top)
    first_area = max(0.0, first["x2"] - first["x1"]) * max(0.0, first["y2"] - first["y1"])
    second_area = max(0.0, second["x2"] - second["x1"]) * max(0.0, second["y2"] - second["y1"])
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def match_image(ground_truth: list[dict[str, Any]], predictions: list[dict[str, Any]], threshold: float = 0.5):
    candidates = sorted(
        (
            (iou(truth, prediction), truth_index, prediction_index)
            for truth_index, truth in enumerate(ground_truth)
            for prediction_index, prediction in enumerate(predictions)
            if int(truth["class_id"]) == int(prediction["class_id"])
        ),
        reverse=True,
    )
    matched_truth: set[int] = set()
    matched_predictions: set[int] = set()
    for overlap, truth_index, prediction_index in candidates:
        if overlap < threshold:
            break
        if truth_index not in matched_truth and prediction_index not in matched_predictions:
            matched_truth.add(truth_index)
            matched_predictions.add(prediction_index)
    return len(matched_truth), len(predictions) - len(matched_predictions), len(ground_truth) - len(matched_truth)


def evaluate_dataset(confidence: float = 0.35, max_images: int = 100) -> dict[str, Any]:
    names = image_choices()[: max(1, int(max_images))]
    if not names:
        raise ValueError("평가할 저장 데이터가 없습니다.")
    client, settings = build_client(confidence)
    totals = Counter()
    per_class = {name: Counter() for name in CLASS_NAMES}
    for filename in names:
        image_path = IMAGE_DIR / filename
        image = cv2.imread(str(image_path))
        if image is None:
            continue
        height, width = image.shape[:2]
        truth = load_boxes(LABEL_DIR / f"{image_path.stem}.txt", width, height)
        predictions = prediction_boxes(infer(client, settings.model_id, image))
        tp, fp, fn = match_image(truth, predictions)
        totals.update({"tp": tp, "fp": fp, "fn": fn})
        for class_id, name in enumerate(CLASS_NAMES):
            class_truth = [box for box in truth if int(box["class_id"]) == class_id]
            class_predictions = [box for box in predictions if int(box["class_id"]) == class_id]
            class_tp, class_fp, class_fn = match_image(class_truth, class_predictions)
            per_class[name].update({"tp": class_tp, "fp": class_fp, "fn": class_fn})

    precision = totals["tp"] / max(1, totals["tp"] + totals["fp"])
    recall = totals["tp"] / max(1, totals["tp"] + totals["fn"])
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    weakest = []
    for name, values in per_class.items():
        support = values["tp"] + values["fn"]
        if support:
            weakest.append({
                "class": name,
                "support": support,
                "recall": values["tp"] / support,
                "false_positives": values["fp"],
            })
    weakest.sort(key=lambda item: (item["recall"], -item["support"]))
    return {
        "model_id": settings.model_id,
        "images": len(names),
        "iou_threshold": 0.5,
        "confidence_threshold": confidence,
        "tp": totals["tp"],
        "fp": totals["fp"],
        "fn": totals["fn"],
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "weakest_classes": weakest[:10],
        "note": "간이 IoU@0.5 평가이며 COCO mAP 계산은 아닙니다.",
    }
