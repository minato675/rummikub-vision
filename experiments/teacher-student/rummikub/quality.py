"""Dataset quality checks used before expensive GPU training."""

from __future__ import annotations

from collections import Counter

from .classes import CLASS_NAMES
from .dataset import IMAGE_DIR, LABEL_DIR, IMAGE_SUFFIXES, ensure_dirs


def audit_dataset() -> dict:
    ensure_dirs()
    images = sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    missing, invalid, counts = [], [], Counter()
    for image in images:
        label = LABEL_DIR / f"{image.stem}.txt"
        if not label.is_file():
            missing.append(image.name)
            continue
        for number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            try:
                values = [float(v) for v in line.split()]
                class_id = int(values[0])
                valid = len(values) == 5 and 0 <= class_id < len(CLASS_NAMES) and all(0 <= v <= 1 for v in values[1:]) and values[3] > 0 and values[4] > 0
            except (ValueError, IndexError):
                valid = False
            if not valid:
                invalid.append(f"{label.name}:{number}")
            else:
                counts[class_id] += 1
    missing_classes = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if not counts[i]]
    rare_classes = [CLASS_NAMES[i] for i in range(len(CLASS_NAMES)) if 0 < counts[i] < 20]
    ready = len(images) >= 100 and not missing and not invalid and not missing_classes
    return {
        "ready": ready,
        "images": len(images),
        "boxes": sum(counts.values()),
        "missing_labels": missing[:30],
        "invalid_labels": invalid[:30],
        "missing_classes": missing_classes,
        "rare_classes_under_20": rare_classes,
        "class_counts": {CLASS_NAMES[i]: counts[i] for i in range(len(CLASS_NAMES))},
    }
