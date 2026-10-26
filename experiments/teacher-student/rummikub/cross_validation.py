"""Leakage-aware 80/20 fold generation for the unified YOLO dataset."""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .classes import CLASS_NAMES
from .dataset import DATA_ROOT, IMAGE_DIR, LABEL_DIR, IMAGE_SUFFIXES


@dataclass
class SampleGroup:
    key: str
    images: list[Path] = field(default_factory=list)
    classes: Counter[int] = field(default_factory=Counter)


def source_group(stem: str) -> str:
    """Keep augmented frames from the same source in the same fold."""
    stem = re.sub(r"^rf\d+_(?:train|valid|val|test)_", "", stem)
    stem = stem.split(".rf.", 1)[0]
    return stem.split("__", 1)[0]


def read_class_counts(path: Path) -> Counter[int]:
    counts: Counter[int] = Counter()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            counts[int(line.split()[0])] += 1
    return counts


def collect_groups() -> list[SampleGroup]:
    groups: dict[str, SampleGroup] = {}
    for image in sorted(p for p in IMAGE_DIR.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES):
        label = LABEL_DIR / f"{image.stem}.txt"
        if not label.is_file():
            raise FileNotFoundError(f"라벨 누락: {image.name}")
        key = source_group(image.stem)
        group = groups.setdefault(key, SampleGroup(key))
        group.images.append(image.resolve())
        group.classes.update(read_class_counts(label))
    return list(groups.values())


def assign_groups(groups: list[SampleGroup], folds: int, seed: int) -> list[list[SampleGroup]]:
    if folds < 2:
        raise ValueError("folds는 2 이상이어야 합니다.")
    if len(groups) < folds:
        raise ValueError(f"독립 이미지 그룹이 최소 {folds}개 필요합니다. 현재 {len(groups)}개입니다.")
    items = list(groups)
    random.Random(seed).shuffle(items)
    items.sort(key=lambda g: (len(g.images), sum(g.classes.values())), reverse=True)
    buckets: list[list[SampleGroup]] = [[] for _ in range(folds)]
    image_totals = [0] * folds
    class_totals = [Counter() for _ in range(folds)]
    target_images = sum(len(g.images) for g in items) / folds
    global_classes = sum((g.classes for g in items), Counter())
    target_classes = {class_id: count / folds for class_id, count in global_classes.items()}

    for group in items:
        def score(index: int) -> tuple[float, int, int]:
            image_penalty = ((image_totals[index] + len(group.images)) / max(target_images, 1)) ** 2
            class_penalty = sum(
                ((class_totals[index][cid] + count) / max(target_classes[cid], 1)) ** 2
                for cid, count in group.classes.items()
            ) / max(len(group.classes), 1)
            return 10 * image_penalty + class_penalty, image_totals[index], index

        chosen = min(range(folds), key=score)
        buckets[chosen].append(group)
        image_totals[chosen] += len(group.images)
        class_totals[chosen].update(group.classes)
    return buckets


def prepare_folds(folds: int = 5, seed: int = 42, output: Path | None = None) -> list[Path]:
    """Create N folds; with five folds every validation portion is approximately 20%."""
    output = output or DATA_ROOT / "folds"
    output.mkdir(parents=True, exist_ok=True)
    groups = collect_groups()
    buckets = assign_groups(groups, folds, seed)
    yaml_paths: list[Path] = []
    manifest: dict[str, object] = {"folds": folds, "seed": seed, "groups": len(groups), "items": []}
    for fold_index, validation_groups in enumerate(buckets, 1):
        validation_keys = {g.key for g in validation_groups}
        train = sorted(i for g in groups if g.key not in validation_keys for i in g.images)
        val = sorted(i for g in validation_groups for i in g.images)
        fold_dir = output / f"fold_{fold_index}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train_txt, val_txt = fold_dir / "train.txt", fold_dir / "val.txt"
        train_txt.write_text("\n".join(p.as_posix() for p in train) + "\n", encoding="utf-8")
        val_txt.write_text("\n".join(p.as_posix() for p in val) + "\n", encoding="utf-8")
        config = {
            "path": str(DATA_ROOT.resolve()),
            "train": str(train_txt.resolve()),
            "val": str(val_txt.resolve()),
            "names": {i: name for i, name in enumerate(CLASS_NAMES)},
        }
        yaml_path = fold_dir / "dataset.yaml"
        yaml_path.write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
        yaml_paths.append(yaml_path)
        manifest["items"].append({"fold": fold_index, "train": len(train), "val": len(val)})
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return yaml_paths
