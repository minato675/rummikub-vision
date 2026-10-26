"""Reviewable auto-labeling with Roboflow, local YOLO26, and point-prompt MobileSAM."""

from __future__ import annotations

import base64
from functools import lru_cache
from pathlib import Path
from statistics import median
from typing import Any

import cv2
import numpy as np
import requests
import torch
from ultralytics import SAM, YOLO

from .classes import CLASS_TO_ID, canonicalize_external_name


ROOT = Path(__file__).resolve().parents[1]
ROBOFLOW_MODEL_ID = "rummikub-p8akb/2016"


def read_roboflow_api_key() -> str:
    for path in (ROOT / ".env", ROOT / "legacy_roboflow" / ".env"):
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise ValueError("ROBOFLOW_API_KEY is missing")


def _box(class_id: int, x1: float, y1: float, x2: float, y2: float, **extra: Any) -> dict[str, Any]:
    return {
        "class_id": int(class_id),
        "x1": int(round(x1)),
        "y1": int(round(y1)),
        "x2": int(round(x2)),
        "y2": int(round(y2)),
        **extra,
    }


def roboflow_auto_boxes(image: np.ndarray, confidence: float = 0.35) -> list[dict[str, Any]]:
    image = np.asarray(image)
    ok, encoded = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("Could not encode image for Roboflow")
    body = base64.b64encode(encoded.tobytes()).decode("ascii")
    response = requests.post(
        f"https://detect.roboflow.com/{ROBOFLOW_MODEL_ID}",
        params={
            "api_key": read_roboflow_api_key(),
            "confidence": round(confidence * 100),
            "format": "json",
        },
        data=body,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        timeout=60,
    )
    response.raise_for_status()
    boxes: list[dict[str, Any]] = []
    for prediction in response.json().get("predictions", []):
        class_name = canonicalize_external_name(str(prediction["class"]))
        width, height = float(prediction["width"]), float(prediction["height"])
        center_x, center_y = float(prediction["x"]), float(prediction["y"])
        boxes.append(
            _box(
                CLASS_TO_ID[class_name],
                center_x - width / 2,
                center_y - height / 2,
                center_x + width / 2,
                center_y + height / 2,
                confidence=float(prediction.get("confidence", 0)),
                source="roboflow",
            )
        )
    return boxes


@lru_cache(maxsize=2)
def _local_model(weights: str) -> YOLO:
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(f"Local YOLO weights not found: {path}")
    return YOLO(str(path))


def local_yolo_auto_boxes(image: np.ndarray, weights: str, confidence: float = 0.35) -> list[dict[str, Any]]:
    result = _local_model(str(Path(weights).resolve())).predict(
        np.asarray(image),
        conf=confidence,
        imgsz=960,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )[0]
    boxes: list[dict[str, Any]] = []
    if result.boxes is None:
        return boxes
    for xyxy, class_id, score in zip(
        result.boxes.xyxy.cpu().tolist(),
        result.boxes.cls.cpu().tolist(),
        result.boxes.conf.cpu().tolist(),
    ):
        class_name = canonicalize_external_name(str(result.names[int(class_id)]))
        boxes.append(
            _box(
                CLASS_TO_ID[class_name],
                *xyxy,
                confidence=float(score),
                source="local_yolo",
            )
        )
    return boxes


@lru_cache(maxsize=1)
def _sam_model() -> SAM:
    return SAM(str(ROOT / "mobile_sam.pt"))


def smart_box_from_point(
    image: np.ndarray,
    point: tuple[int, int],
    class_id: int,
    reference_boxes: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    image = np.asarray(image)
    height, width = image.shape[:2]
    x, y = point
    if reference_boxes:
        half_width = median(abs(box["x2"] - box["x1"]) for box in reference_boxes) * 0.55
        half_height = median(abs(box["y2"] - box["y1"]) for box in reference_boxes) * 0.55
    else:
        half_width, half_height = width * 0.07, height * 0.07
    prompt_box = [
        max(0, x - half_width),
        max(0, y - half_height),
        min(width - 1, x + half_width),
        min(height - 1, y + half_height),
    ]
    results = _sam_model().predict(
        source=image,
        bboxes=prompt_box,
        device=0 if torch.cuda.is_available() else "cpu",
        verbose=False,
    )
    candidates: list[tuple[float, list[float]]] = []
    for result in results:
        if result.boxes is None:
            continue
        for coords in result.boxes.xyxy.cpu().tolist():
            x1, y1, x2, y2 = coords
            if x1 <= x <= x2 and y1 <= y <= y2:
                area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
                image_ratio = area / max(1, width * height)
                if 0.0002 <= image_ratio <= 0.5:
                    candidates.append((area, coords))
    if not candidates:
        raise RuntimeError("MobileSAM could not find an object boundary at this point")
    _, coords = min(candidates, key=lambda item: abs(item[0] - (half_width * 2) * (half_height * 2)))
    return _box(class_id, *coords, source="mobile_sam")
