"""Run the hosted Roboflow Rummikub detector on an image."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from inference_sdk import InferenceHTTPClient
from PIL import Image, ImageDraw, ImageFont


DEFAULT_MODEL_ID = "rummikub-p8akb/2016"
DEFAULT_API_URL = "https://serverless.roboflow.com"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect Rummikub tiles using the Roboflow hosted model."
    )
    parser.add_argument("image", type=Path, help="Path to an input image")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/result.jpg"),
        help="Annotated image path (default: outputs/result.jpg)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=Path("outputs/result.json"),
        help="Raw prediction JSON path (default: outputs/result.json)",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.4,
        help="Only draw detections at or above this confidence (default: 0.4)",
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID),
        help=f"Roboflow model ID (default: {DEFAULT_MODEL_ID})",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.image.is_file():
        raise ValueError(f"Input image does not exist: {args.image}")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")


def create_client() -> InferenceHTTPClient:
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError(
            "ROBOFLOW_API_KEY is missing. Copy .env.example to .env and add your key."
        )
    return InferenceHTTPClient(
        api_url=os.getenv("ROBOFLOW_API_URL", DEFAULT_API_URL),
        api_key=api_key,
    )


def annotate_image(
    image_path: Path,
    predictions: list[dict[str, Any]],
    output_path: Path,
    confidence_threshold: float,
) -> int:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    drawn = 0

    for prediction in predictions:
        confidence = float(prediction.get("confidence", 0))
        if confidence < confidence_threshold:
            continue

        x = float(prediction["x"])
        y = float(prediction["y"])
        width = float(prediction["width"])
        height = float(prediction["height"])
        left, top = x - width / 2, y - height / 2
        right, bottom = x + width / 2, y + height / 2
        label = f'{prediction.get("class", "unknown")} {confidence:.0%}'

        draw.rectangle((left, top, right, bottom), outline="#00ff88", width=3)
        text_box = draw.textbbox((left, top), label, font=font)
        text_height = text_box[3] - text_box[1]
        label_top = max(0, top - text_height - 6)
        label_box = draw.textbbox((left + 3, label_top + 3), label, font=font)
        draw.rectangle(
            (left, label_top, label_box[2] + 3, label_box[3] + 3), fill="#00ff88"
        )
        draw.text((left + 3, label_top + 3), label, fill="black", font=font)
        drawn += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return drawn


def main() -> int:
    load_dotenv()
    args = parse_args()

    try:
        validate_args(args)
        client = create_client()
        result = client.infer(str(args.image.resolve()), model_id=args.model_id)

        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        count = annotate_image(
            args.image,
            result.get("predictions", []),
            args.output,
            args.confidence,
        )
    except (ValueError, OSError, KeyError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # The SDK raises several transport-specific exceptions.
        print(f"Roboflow inference failed: {exc}", file=sys.stderr)
        return 2

    print(f"Detected {count} tile(s).")
    print(f"Annotated image: {args.output.resolve()}")
    print(f"Prediction JSON: {args.json_output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
