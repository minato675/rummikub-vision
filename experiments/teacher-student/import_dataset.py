"""Import/merge a YOLO dataset into data/images/all and data/labels/all."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from rummikub.dataset import IMAGE_DIR, LABEL_DIR, IMAGE_SUFFIXES, ensure_dirs


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("source", type=Path, help="Folder containing images/all and labels/all")
    p.add_argument("--prefix", default="imported", help="Collision-safe filename prefix")
    args = p.parse_args()
    source_images, source_labels = args.source / "images" / "all", args.source / "labels" / "all"
    if not source_images.is_dir() or not source_labels.is_dir():
        raise FileNotFoundError("source 아래에 images/all 및 labels/all이 필요합니다.")
    ensure_dirs(); copied = 0
    for image in sorted(p for p in source_images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = source_labels / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"라벨 누락: {image.name}")
        stem = image.stem
        target_image = IMAGE_DIR / image.name
        target_label = LABEL_DIR / label.name
        if target_image.exists() or target_label.exists():
            stem = f"{args.prefix}_{stem}"
            target_image = IMAGE_DIR / f"{stem}{image.suffix.lower()}"
            target_label = LABEL_DIR / f"{stem}.txt"
        shutil.copy2(image, target_image); shutil.copy2(label, target_label); copied += 1
    print(f"{copied}개 이미지/라벨 쌍을 가져왔습니다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
