"""Train a YOLO26 student with group-safe 80/20 cross-validation."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from rummikub.cross_validation import prepare_folds
from rummikub.quality import audit_dataset

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolo26n.pt", help="yolo26n.pt or a previous best.pt")
    p.add_argument(
        "--distill-model",
        default=None,
        help="Optional teacher checkpoint used for Ultralytics feature distillation",
    )
    p.add_argument(
        "--distill-weight",
        type=float,
        default=6.0,
        help="Feature-distillation loss weight (Ultralytics 'dis')",
    )
    p.add_argument("--optimizer", default="auto")
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--batch", type=float, default=-1)
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--fold", default="1", help="1..N or all")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--target-map50", type=float, default=0.90)
    p.add_argument(
        "--run-name",
        default="fold",
        help="Output prefix used to preserve separate experiments",
    )
    p.add_argument("--prepare-only", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    audit = audit_dataset()
    if not audit["ready"]:
        raise RuntimeError("데이터 품질 검사 실패: " + json.dumps(audit, ensure_ascii=False))
    fold_yamls = prepare_folds(args.folds, args.seed)
    if args.prepare_only:
        print(json.dumps({"audit": audit, "folds": [str(p) for p in fold_yamls]}, ensure_ascii=False, indent=2))
        return 0

    import torch
    from ultralytics import YOLO

    device = args.device if args.device != "auto" else ("0" if torch.cuda.is_available() else "cpu")
    batch = int(args.batch) if float(args.batch).is_integer() else args.batch
    if args.fold == "all":
        selected = list(enumerate(fold_yamls, 1))
    else:
        fold_number = int(args.fold)
        if not 1 <= fold_number <= len(fold_yamls):
            raise ValueError(f"fold는 1~{len(fold_yamls)} 또는 all이어야 합니다.")
        selected = [(fold_number, fold_yamls[fold_number - 1])]

    project = ROOT / "runs" / "student_cv"
    rows = []
    for fold_number, data_yaml in selected:
        print(f"\n===== Fold {fold_number}/{args.folds} =====", flush=True)
        model = YOLO(args.model)
        model.train(
            data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=batch,
            device=device, workers=args.workers, patience=args.patience,
            project=str(project), name=f"{args.run_name}_fold_{fold_number}", exist_ok=True,
            pretrained=True, seed=args.seed + fold_number, deterministic=True,
            close_mosaic=10, plots=True, amp=True,
            distill_model=args.distill_model,
            dis=args.distill_weight,
            optimizer=args.optimizer,
            lr0=args.lr0,
        )
        metrics = model.val(data=str(data_yaml), split="val", device=device, imgsz=args.imgsz)
        row = {
            "fold": fold_number,
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50), "map50_95": float(metrics.box.map),
            "passed_90": float(metrics.box.map50) >= args.target_map50,
            "best_weights": str(Path(model.trainer.best).resolve()),
        }
        rows.append(row)
        print("RESULT " + json.dumps(row, ensure_ascii=False), flush=True)

    project.mkdir(parents=True, exist_ok=True)
    csv_path = project / f"{args.run_name}_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    mean = {key: sum(float(row[key]) for row in rows) / len(rows) for key in ("precision", "recall", "map50", "map50_95")}
    summary = {
        "target_map50": args.target_map50,
        "target_passed": mean["map50"] >= args.target_map50,
        "mean": mean, "results": rows, "audit": audit,
        "student_model": str(args.model),
        "distill_model": str(args.distill_model) if args.distill_model else None,
        "distill_weight": args.distill_weight,
        "optimizer": args.optimizer,
        "lr0": args.lr0,
        "note": "단일 fold 실행 시 mean은 해당 fold 결과입니다. 최종 보고에는 --fold all을 사용하세요.",
    }
    summary["run_name"] = args.run_name
    experiment_summary = project / f"{args.run_name}_summary.json"
    experiment_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    # Studio always exposes the most recently completed experiment here.
    (project / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
