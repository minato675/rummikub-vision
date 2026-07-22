"""Import the licensed Roboflow Universe Rummikub annotations into local YOLO data."""

from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import requests
import yaml

from rummikub.annotations import DATA_ROOT, ensure_dataset_dirs
from rummikub.classes import CLASS_NAMES, CLASS_TO_ID, canonicalize_external_name


ROOT = Path(__file__).resolve().parent
WORKSPACE = "arthurvanmeerbeeck-gmail-com"
PROJECT = "rummikub-p8akb"
DEFAULT_VERSION = 2017
FORMAT = "yolov8"
SOURCE_URL = f"https://universe.roboflow.com/{WORKSPACE}/{PROJECT}"


def read_api_key() -> str:
    candidates = [ROOT / ".env", ROOT / "legacy_roboflow" / ".env"]
    for path in candidates:
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.startswith("ROBOFLOW_API_KEY="):
                value = line.split("=", 1)[1].strip().strip('"').strip("'")
                if value:
                    return value
    raise ValueError("ROBOFLOW_API_KEY was not found in .env or legacy_roboflow/.env")


def request_export(api_key: str, version: int) -> tuple[str, dict]:
    endpoint = f"https://api.roboflow.com/{WORKSPACE}/{PROJECT}/{version}/{FORMAT}"
    response = requests.get(endpoint, params={"api_key": api_key}, timeout=60)
    response.raise_for_status()
    payload = response.json()
    link = payload.get("export", {}).get("link")
    if not link:
        raise RuntimeError(f"Roboflow did not return an export link: {payload.get('error', 'unknown error')}")
    return link, payload


def download_zip(link: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(link, stream=True, timeout=180) as response:
        response.raise_for_status()
        with destination.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)


def safe_extract(archive: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    destination_root = destination.resolve()
    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            target = (destination / member.filename).resolve()
            if destination_root != target and destination_root not in target.parents:
                raise ValueError(f"Unsafe archive path: {member.filename}")
        bundle.extractall(destination)


def load_export_names(extracted: Path) -> list[str]:
    yaml_files = list(extracted.rglob("data.yaml"))
    if not yaml_files:
        raise FileNotFoundError("Export does not contain data.yaml")
    config = yaml.safe_load(yaml_files[0].read_text(encoding="utf-8"))
    names = config.get("names")
    if isinstance(names, dict):
        return [str(names[index] if index in names else names[str(index)]) for index in range(len(names))]
    return [str(name) for name in names]


def to_detection_line(parts: list[str], target_class_id: int) -> str:
    values = [float(value) for value in parts[1:]]
    if len(values) == 4:
        center_x, center_y, width, height = values
    elif len(values) >= 6 and len(values) % 2 == 0:
        x_values = values[0::2]
        y_values = values[1::2]
        x1, x2 = min(x_values), max(x_values)
        y1, y2 = min(y_values), max(y_values)
        center_x, center_y = (x1 + x2) / 2, (y1 + y2) / 2
        width, height = x2 - x1, y2 - y1
    else:
        raise ValueError(f"Unsupported Roboflow label with {len(parts)} columns")
    return f"{target_class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}"


def import_dataset(
    extracted: Path,
    source_names: list[str],
    version: int,
    output_root: Path = DATA_ROOT,
) -> dict:
    ensure_dataset_dirs(output_root)
    id_map = {index: CLASS_TO_ID[canonicalize_external_name(name)] for index, name in enumerate(source_names)}
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "test"}
    counts = {"train": {"images": 0, "boxes": 0}, "val": {"images": 0, "boxes": 0}, "test": {"images": 0, "boxes": 0}}

    for source_split, target_split in split_map.items():
        candidates = [path for path in extracted.rglob(source_split) if path.is_dir() and (path / "images").is_dir()]
        for split_root in candidates:
            image_dir, label_dir = split_root / "images", split_root / "labels"
            for image_path in image_dir.iterdir():
                if image_path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
                    continue
                target_stem = f"rf{version}_{source_split}_{image_path.stem}"
                target_image = output_root / "images" / target_split / f"{target_stem}{image_path.suffix.lower()}"
                target_label = output_root / "labels" / target_split / f"{target_stem}.txt"
                source_label = label_dir / f"{image_path.stem}.txt"
                converted: list[str] = []
                if source_label.is_file():
                    for line in source_label.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        parts = line.split()
                        source_id = int(parts[0])
                        converted.append(to_detection_line(parts, id_map[source_id]))
                shutil.copy2(image_path, target_image)
                target_label.write_text("\n".join(converted) + ("\n" if converted else ""), encoding="utf-8")
                counts[target_split]["images"] += 1
                counts[target_split]["boxes"] += len(converted)
    return {"counts": counts, "class_mapping": {source_names[index]: CLASS_NAMES[target] for index, target in id_map.items()}}


def main() -> int:
    parser = argparse.ArgumentParser(description="Import Roboflow Rummikub YOLO labels")
    parser.add_argument("--version", type=int, default=DEFAULT_VERSION)
    args = parser.parse_args()
    temp_root = ROOT / "tmp" / f"roboflow_{PROJECT}_{args.version}"
    archive = ROOT / "tmp" / f"roboflow_{PROJECT}_{args.version}.zip"
    api_key = read_api_key()
    print(f"Requesting {PROJECT}/{args.version} ({FORMAT}) export...")
    link, metadata = request_export(api_key, args.version)
    download_zip(link, archive)
    safe_extract(archive, temp_root)
    source_names = load_export_names(temp_root)
    result = import_dataset(temp_root, source_names, args.version)
    manifest = {
        "source": SOURCE_URL,
        "workspace": WORKSPACE,
        "project": PROJECT,
        "version": args.version,
        "format": FORMAT,
        "license": "CC BY 4.0",
        "imported_at": datetime.now(timezone.utc).isoformat(),
        "counts": result["counts"],
        "class_mapping": result["class_mapping"],
        "preprocessing": metadata.get("version", {}).get("preprocessing"),
        "augmentation": metadata.get("version", {}).get("augmentation"),
    }
    manifest_path = ROOT / "data" / "roboflow_import_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["counts"], ensure_ascii=False, indent=2))
    print(f"Provenance: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
