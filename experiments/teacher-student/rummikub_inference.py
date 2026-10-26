"""Shared client and rendering helpers for the local Roboflow model."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv
from inference_sdk import InferenceConfiguration, InferenceHTTPClient

from rummikub.visuals import draw_box_label


ROOT = Path(__file__).resolve().parent
DEFAULT_MODEL_ID = "rummikub-p8akb/2016"
DEFAULT_SERVER_URL = "http://127.0.0.1:9001"


@dataclass(frozen=True)
class Settings:
    api_key: str
    model_id: str
    server_url: str


def load_settings(require_api_key: bool = True) -> Settings:
    load_dotenv(ROOT / ".env")
    api_key = os.getenv("ROBOFLOW_API_KEY", "").strip()
    if require_api_key and not api_key:
        raise ValueError(f"ROBOFLOW_API_KEY is missing. Create {ROOT / '.env'} from .env.example")
    return Settings(
        api_key=api_key,
        model_id=os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID).strip() or DEFAULT_MODEL_ID,
        server_url=os.getenv("ROBOFLOW_LOCAL_URL", DEFAULT_SERVER_URL).strip().rstrip("/") or DEFAULT_SERVER_URL,
    )


def build_client(confidence: float = 0.35) -> tuple[InferenceHTTPClient, Settings]:
    settings = load_settings()
    client = InferenceHTTPClient(api_url=settings.server_url, api_key=settings.api_key)
    client.configure(InferenceConfiguration(confidence_threshold=confidence))
    return client, settings


def infer(client: InferenceHTTPClient, model_id: str, image: np.ndarray) -> dict[str, Any]:
    result = client.infer(image, model_id=model_id)
    if isinstance(result, list):
        if not result:
            return {"predictions": []}
        result = result[0]
    if not isinstance(result, dict):
        raise TypeError(f"Unexpected inference response: {type(result).__name__}")
    return result


def draw_predictions(image: np.ndarray, result: dict[str, Any]) -> np.ndarray:
    canvas = np.asarray(image).copy()
    for prediction in result.get("predictions", []):
        center_x = float(prediction["x"])
        center_y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        x1, y1 = int(center_x - width / 2), int(center_y - height / 2)
        x2, y2 = int(center_x + width / 2), int(center_y + height / 2)
        confidence = float(prediction.get("confidence", 0))
        class_name = str(prediction.get("class", "unknown"))
        label = f"{class_name}  {confidence:.0%}"
        draw_box_label(canvas, (x1, y1, x2, y2), label, class_name, channel_order="BGR")
    return canvas
