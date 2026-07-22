"""Validate YOLO detection image/label pairs before training."""

from __future__ import annotations

from collections import Counter
from pathlib import Path

from rummikub.annotations import DATA_ROOT, SPLITS
from rummikub.classes import CLASS_NAMES


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate(root: Path = DATA_ROOT) -> list[str]:
    errors: list[str] = []
    counts: Counter[int] = Counter()
    for split in SPLITS:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = {path.stem: path for path in image_dir.glob("*") if path.suffix.lower() in IMAGE_SUFFIXES}
        labels = {path.stem: path for path in label_dir.glob("*.txt")}
        for stem in sorted(images.keys() - labels.keys()):
            errors.append(f"{split}: missing label for {images[stem].name}")
        for stem in sorted(labels.keys() - images.keys()):
            errors.append(f"{split}: missing image for {labels[stem].name}")
        for label_path in labels.values():
            for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    errors.append(f"{label_path}:{line_number}: expected 5 columns")
                    continue
                try:
                    class_id = int(parts[0])
                    values = [float(value) for value in parts[1:]]
                except ValueError:
                    errors.append(f"{label_path}:{line_number}: non-numeric value")
                    continue
                if not 0 <= class_id < len(CLASS_NAMES):
                    errors.append(f"{label_path}:{line_number}: invalid class {class_id}")
                elif not all(0 <= value <= 1 for value in values):
                    errors.append(f"{label_path}:{line_number}: coordinates must be in [0,1]")
                else:
                    counts[class_id] += 1
    if errors:
        return errors
    print("Dataset validation passed.")
    print(f"Labeled classes: {len(counts)}/{len(CLASS_NAMES)}")
    for class_id, count in sorted(counts.items()):
        print(f"  {CLASS_NAMES[class_id]}: {count}")
    return []


if __name__ == "__main__":
    found = validate()
    if found:
        print("Dataset validation failed:")
        for error in found:
            print(f"  - {error}")
        raise SystemExit(1)
