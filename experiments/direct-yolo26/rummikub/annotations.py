"""YOLO annotation persistence and visualization helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np

from .classes import CLASS_NAMES


DATA_ROOT = Path(__file__).resolve().parents[1] / "data"
SPLITS = ("train", "val", "test")


def ensure_dataset_dirs(root: Path = DATA_ROOT) -> None:
    for split in SPLITS:
        (root / "images" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)


def normalize_box(box: dict[str, Any], width: int, height: int) -> str:
    x1, x2 = sorted((float(box["x1"]), float(box["x2"])))
    y1, y2 = sorted((float(box["y1"]), float(box["y2"])))
    x1, x2 = max(0, x1), min(width - 1, x2)
    y1, y2 = max(0, y1), min(height - 1, y2)
    box_width, box_height = x2 - x1, y2 - y1
    if box_width < 2 or box_height < 2:
        raise ValueError("Bounding box must be at least 2x2 pixels")
    center_x = (x1 + x2) / 2 / width
    center_y = (y1 + y2) / 2 / height
    return (
        f'{int(box["class_id"])} {center_x:.6f} {center_y:.6f} '
        f'{box_width / width:.6f} {box_height / height:.6f}'
    )


def draw_boxes(
    image: np.ndarray, boxes: list[dict[str, Any]], pending: tuple[int, int] | None = None
) -> np.ndarray:
    canvas = np.asarray(image).copy()
    if canvas.ndim == 2:
        canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
    for box in boxes:
        x1, x2 = sorted((int(box["x1"]), int(box["x2"])))
        y1, y2 = sorted((int(box["y1"]), int(box["y2"])))
        class_id = int(box["class_id"])
        label = CLASS_NAMES[class_id]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 230, 130), 3)
        cv2.rectangle(canvas, (x1, max(0, y1 - 24)), (x1 + 10 + len(label) * 9, y1), (0, 230, 130), -1)
        cv2.putText(canvas, label, (x1 + 4, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    if pending is not None:
        cv2.drawMarker(canvas, pending, (255, 200, 0), cv2.MARKER_CROSS, 24, 3)
    return canvas


def save_yolo_sample(
    image: np.ndarray,
    boxes: list[dict[str, Any]],
    split: str,
    root: Path = DATA_ROOT,
) -> tuple[Path, Path]:
    if split not in SPLITS:
        raise ValueError(f"Unknown split: {split}")
    if image is None:
        raise ValueError("No image was provided")
    ensure_dataset_dirs(root)
    image = np.asarray(image)
    height, width = image.shape[:2]
    stem = f"rummikub_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    image_path = root / "images" / split / f"{stem}.jpg"
    label_path = root / "labels" / split / f"{stem}.txt"
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(str(image_path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"Could not save image: {image_path}")
    lines = [normalize_box(box, width, height) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return image_path, label_path


def dataset_summary(root: Path = DATA_ROOT) -> str:
    ensure_dataset_dirs(root)
    lines = ["### 데이터셋 현황", "", "| 분할 | 이미지 | 라벨 박스 |", "|---|---:|---:|"]
    grand_classes: Counter[int] = Counter()
    for split in SPLITS:
        image_count = len(list((root / "images" / split).glob("*.*")))
        box_count = 0
        for label_file in (root / "labels" / split).glob("*.txt"):
            for line in label_file.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                class_id = int(line.split()[0])
                grand_classes[class_id] += 1
                box_count += 1
        lines.append(f"| {split} | {image_count} | {box_count} |")
    if grand_classes:
        missing = [name for index, name in enumerate(CLASS_NAMES) if grand_classes[index] == 0]
        lines.extend(["", f"라벨이 있는 클래스: **{len(grand_classes)}/53**"])
        if missing:
            lines.append(f"라벨이 아직 없는 클래스: `{', '.join(missing)}`")
    else:
        lines.extend(["", "아직 저장된 라벨이 없습니다."])
    return "\n".join(lines)
