"""Persistence and export helpers for reviewed YOLO annotations."""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import yaml

from .classes import CLASS_NAMES


ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = ROOT / "data"
IMAGE_DIR = DATA_ROOT / "images" / "all"
LABEL_DIR = DATA_ROOT / "labels" / "all"
EXPORT_DIR = ROOT / "exports"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def ensure_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    LABEL_DIR.mkdir(parents=True, exist_ok=True)
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)


def image_choices() -> list[str]:
    ensure_dirs()
    return sorted(path.name for path in IMAGE_DIR.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)


def normalize_box(box: dict[str, Any], width: int, height: int) -> str:
    x1, x2 = sorted((max(0.0, float(box["x1"])), min(width - 1.0, float(box["x2"]))))
    y1, y2 = sorted((max(0.0, float(box["y1"])), min(height - 1.0, float(box["y2"]))))
    if x2 - x1 < 2 or y2 - y1 < 2:
        raise ValueError("박스는 최소 2×2 픽셀이어야 합니다.")
    return (
        f'{int(box["class_id"])} {(x1 + x2) / 2 / width:.6f} {(y1 + y2) / 2 / height:.6f} '
        f'{(x2 - x1) / width:.6f} {(y2 - y1) / height:.6f}'
    )


def load_boxes(label_path: Path, width: int, height: int) -> list[dict[str, Any]]:
    boxes: list[dict[str, Any]] = []
    if not label_path.is_file():
        return boxes
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, cx, cy, bw, bh = map(float, line.split())
        boxes.append({
            "class_id": int(class_id),
            "x1": (cx - bw / 2) * width,
            "y1": (cy - bh / 2) * height,
            "x2": (cx + bw / 2) * width,
            "y2": (cy + bh / 2) * height,
            "source": "dataset",
        })
    return boxes


def save_sample(image: np.ndarray, boxes: list[dict[str, Any]]) -> tuple[Path, Path]:
    ensure_dirs()
    height, width = image.shape[:2]
    stem = f"rummikub_{datetime.now():%Y%m%d_%H%M%S}_{uuid4().hex[:8]}"
    image_path, label_path = IMAGE_DIR / f"{stem}.jpg", LABEL_DIR / f"{stem}.txt"
    if not cv2.imwrite(str(image_path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95]):
        raise OSError(f"이미지 저장 실패: {image_path}")
    lines = [normalize_box(box, width, height) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return image_path, label_path


def overwrite_labels(label_path: Path, image: np.ndarray, boxes: list[dict[str, Any]]) -> None:
    height, width = image.shape[:2]
    lines = [normalize_box(box, width, height) for box in boxes]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def summary() -> str:
    ensure_dirs()
    counts: Counter[int] = Counter()
    box_count = 0
    for label in LABEL_DIR.glob("*.txt"):
        for line in label.read_text(encoding="utf-8").splitlines():
            if line.strip():
                counts[int(line.split()[0])] += 1
                box_count += 1
    return (
        "### 추가 라벨 데이터\n\n"
        f"- 이미지: **{len(image_choices())}장**\n"
        f"- 박스: **{box_count}개**\n"
        f"- 포함 클래스: **{len(counts)}/53**"
    )


def export_roboflow_zip() -> Path:
    ensure_dirs()
    if not image_choices():
        raise ValueError("내보낼 라벨 데이터가 없습니다.")
    output = EXPORT_DIR / f"rummikub_reviewed_{datetime.now():%Y%m%d_%H%M%S}.zip"
    with tempfile.TemporaryDirectory(prefix="rummikub_export_") as temporary:
        bundle = Path(temporary)
        train_images, train_labels = bundle / "train" / "images", bundle / "train" / "labels"
        train_images.mkdir(parents=True)
        train_labels.mkdir(parents=True)
        for image in IMAGE_DIR.iterdir():
            if image.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            label = LABEL_DIR / f"{image.stem}.txt"
            if not label.is_file():
                raise FileNotFoundError(f"라벨 누락: {image.name}")
            shutil.copy2(image, train_images / image.name)
            shutil.copy2(label, train_labels / label.name)
        config = {"path": ".", "train": "train/images", "val": "train/images", "names": CLASS_NAMES}
        (bundle / "data.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
            for path in bundle.rglob("*"):
                if path.is_file():
                    archive.write(path, path.relative_to(bundle))
    return output
