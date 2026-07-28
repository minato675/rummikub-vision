"""Safely snapshot and stop an active Ultralytics job after a target epoch."""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import subprocess
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def latest_row(results_csv: Path) -> dict[str, str] | None:
    if not results_csv.is_file():
        return None
    with results_csv.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-1] if rows else None


def process_exists(pid: int) -> bool:
    result = subprocess.run(
        ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
        capture_output=True,
        text=True,
    )
    return f'"{pid}"' in result.stdout


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--target", type=int, required=True)
    parser.add_argument("--fold-dir", type=Path, required=True)
    args = parser.parse_args()

    fold_dir = args.fold_dir.resolve()
    if ROOT not in fold_dir.parents:
        raise ValueError(f"fold-dir must stay inside {ROOT}: {fold_dir}")
    results_csv = fold_dir / "results.csv"
    weights_dir = fold_dir / "weights"
    last_pt = weights_dir / "last.pt"
    snapshot = weights_dir / f"epoch_{args.target}.pt"
    metrics_path = fold_dir / f"epoch_{args.target}_metrics.json"
    watcher_state = ROOT / "runs" / "epoch_stop_watcher.json"

    watcher_state.parent.mkdir(parents=True, exist_ok=True)
    watcher_state.write_text(
        json.dumps(
            {"status": "watching", "pid": args.pid, "target_epoch": args.target},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Watching PID {args.pid} for epoch {args.target}", flush=True)

    while process_exists(args.pid):
        row = latest_row(results_csv)
        if row and int(float(row["epoch"])) >= args.target:
            # Ultralytics writes results.csv immediately before updating last.pt.
            deadline = time.time() + 180
            while time.time() < deadline:
                if (
                    last_pt.is_file()
                    and last_pt.stat().st_size > 1_000_000
                    and last_pt.stat().st_mtime_ns >= results_csv.stat().st_mtime_ns
                ):
                    size = last_pt.stat().st_size
                    time.sleep(3)
                    if last_pt.stat().st_size == size:
                        break
                time.sleep(1)
            else:
                raise TimeoutError("Epoch row appeared, but last.pt was not safely updated.")

            temporary = snapshot.with_suffix(".tmp")
            shutil.copy2(last_pt, temporary)
            os.replace(temporary, snapshot)
            metrics = {
                "stopped_at": datetime.now().isoformat(timespec="seconds"),
                "epoch": args.target,
                "precision": float(row["metrics/precision(B)"]),
                "recall": float(row["metrics/recall(B)"]),
                "map50": float(row["metrics/mAP50(B)"]),
                "map50_95": float(row["metrics/mAP50-95(B)"]),
                "snapshot": str(snapshot),
                "best_checkpoint": str(weights_dir / "best.pt"),
            }
            metrics_path.write_text(
                json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            subprocess.run(
                ["taskkill", "/PID", str(args.pid), "/T", "/F"],
                capture_output=True,
                text=True,
            )
            watcher_state.write_text(
                json.dumps({"status": "stopped_and_saved", **metrics}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            training_state = ROOT / "runs" / "training_job.json"
            if training_state.is_file():
                state = json.loads(training_state.read_text(encoding="utf-8"))
                state.update(
                    {
                        "status": "stopped_at_epoch",
                        "stopped_epoch": args.target,
                        "snapshot": str(snapshot),
                        "metrics": metrics,
                    }
                )
                training_state.write_text(
                    json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8"
                )
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            return 0
        time.sleep(2)

    watcher_state.write_text(
        json.dumps(
            {"status": "training_process_ended_early", "pid": args.pid},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
