"""Fine-tune Ultralytics YOLO26n on the local Rummikub dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLO26n for Rummikub tiles")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.yaml")
    parser.add_argument("--model", default="yolo26n.pt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--batch", type=float, default=-1, help="-1 uses automatic GPU batch sizing")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--name", default="rummikub_yolo26n")
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume:
        YOLO(str(args.resume)).train(resume=True)
        return 0
    model = YOLO(args.model)
    model.train(
        data=str(args.data.resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=str(ROOT / "runs" / "detect"),
        name=args.name,
        pretrained=True,
        cache=False,
        seed=42,
        deterministic=True,
        close_mosaic=10,
        plots=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
