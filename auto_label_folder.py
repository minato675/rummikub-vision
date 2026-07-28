"""Create review-required YOLO drafts for every image in a folder."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import cv2

from rummikub.dataset import IMAGE_DIR, LABEL_DIR, IMAGE_SUFFIXES, ensure_dirs, normalize_box
from rummikub.evaluation import prediction_boxes
from rummikub_inference import build_client, infer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--prefix", default="teacher")
    args = parser.parse_args()
    if not args.source.is_dir():
        raise NotADirectoryError(args.source)
    ensure_dirs()
    client, settings = build_client(args.confidence)
    images = sorted(p for p in args.source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    for index, source in enumerate(images, 1):
        image = cv2.imread(str(source))
        if image is None:
            print(f"SKIP 읽기 실패: {source.name}")
            continue
        stem = f"{args.prefix}_{source.stem}"
        target_image = IMAGE_DIR / f"{stem}{source.suffix.lower()}"
        target_label = LABEL_DIR / f"{stem}.txt"
        suffix = 1
        while target_image.exists() or target_label.exists():
            target_image = IMAGE_DIR / f"{stem}_{suffix}{source.suffix.lower()}"
            target_label = LABEL_DIR / f"{stem}_{suffix}.txt"
            suffix += 1
        result = infer(client, settings.model_id, image)
        boxes = prediction_boxes(result)
        height, width = image.shape[:2]
        lines = [normalize_box(box, width, height) for box in boxes]
        shutil.copy2(source, target_image)
        target_label.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        print(f"[{index}/{len(images)}] {target_image.name}: {len(boxes)} boxes", flush=True)
    print("완료. Studio에서 모든 자동 라벨을 사람이 검수해야 합니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
