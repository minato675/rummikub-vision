"""Run the cached Roboflow model on one local image."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from rummikub_inference import build_client, draw_predictions, infer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("--output", type=Path, default=Path("outputs/inference.jpg"))
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()
    image = cv2.imread(str(args.image.resolve()))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    client, settings = build_client(args.confidence)
    result = infer(client, settings.model_id, image)
    annotated = draw_predictions(image, result)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), annotated):
        raise OSError(f"Could not save: {output}")
    print(f"Model: {settings.model_id}")
    print(f"Local server: {settings.server_url}")
    print(f"Predictions: {len(result.get('predictions', []))}")
    print(f"Output: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

