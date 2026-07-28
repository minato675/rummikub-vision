"""Wait for an existing training PID, then launch a configured training job."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from training_control import _process_exists, start_training

ROOT = Path(__file__).resolve().parent
QUEUE_STATE = ROOT / "runs" / "training_queue.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wait-pid", type=int, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--distill-model", default=None)
    parser.add_argument("--distill-weight", type=float, default=6.0)
    parser.add_argument("--optimizer", default="auto")
    parser.add_argument("--lr0", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=float, default=14)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--fold", default="all")
    parser.add_argument("--target-map50", type=float, default=0.90)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--poll-seconds", type=int, default=15)
    return parser.parse_args()


def write_state(status: str, args: argparse.Namespace, message: str = "") -> None:
    QUEUE_STATE.parent.mkdir(parents=True, exist_ok=True)
    QUEUE_STATE.write_text(
        json.dumps(
            {
                "status": status,
                "waiting_for_pid": args.wait_pid,
                "student_model": str(Path(args.model).resolve()),
                "distill_model": (
                    str(Path(args.distill_model).resolve())
                    if args.distill_model
                    else None
                ),
                "distill_weight": args.distill_weight,
                "optimizer": args.optimizer,
                "lr0": args.lr0,
                "epochs": args.epochs,
                "imgsz": args.imgsz,
                "batch": args.batch,
                "patience": args.patience,
                "fold": args.fold,
                "run_name": args.run_name,
                "message": message,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()
    write_state("waiting", args)
    while _process_exists(args.wait_pid):
        time.sleep(max(2, args.poll_seconds))

    message = start_training(
        args.model,
        args.epochs,
        args.imgsz,
        args.batch,
        args.patience,
        args.fold,
        args.target_map50,
        args.run_name,
        args.distill_model,
        args.distill_weight,
        args.optimizer,
        args.lr0,
    )
    write_state("launched", args, message)
    print(message, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
