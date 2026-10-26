"""Ultralytics YOLO26 inference adapter."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from .rules import Tile


@lru_cache(maxsize=2)
def load_model(weights: str) -> YOLO:
    path = Path(weights)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {path}. Train first or select runs/.../weights/best.pt"
        )
    return YOLO(str(path))


def detect_tiles(
    image: np.ndarray,
    weights: str,
    confidence: float = 0.35,
    image_size: int = 960,
) -> tuple[list[Tile], np.ndarray]:
    if image is None:
        raise ValueError("Image is required")
    model = load_model(str(Path(weights).resolve()))
    result = model.predict(
        source=np.asarray(image),
        conf=confidence,
        imgsz=image_size,
        verbose=False,
    )[0]
    names = result.names
    tiles: list[Tile] = []
    if result.boxes is not None:
        for xyxy, class_id, score in zip(
            result.boxes.xyxy.cpu().tolist(),
            result.boxes.cls.cpu().tolist(),
            result.boxes.conf.cpu().tolist(),
        ):
            x1, y1, x2, y2 = xyxy
            tiles.append(
                Tile(
                    class_name=str(names[int(class_id)]),
                    confidence=float(score),
                    x1=float(x1),
                    y1=float(y1),
                    x2=float(x2),
                    y2=float(y2),
                )
            )
    plotted_bgr = result.plot()
    return tiles, cv2.cvtColor(plotted_bgr, cv2.COLOR_BGR2RGB)
