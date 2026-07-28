"""Non-blocking local training process control for Gradio."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
STATE = ROOT / "runs" / "training_job.json"
LOG = ROOT / "runs" / "training.log"


def training_python() -> Path:
    configured = os.getenv("RUMMIKUB_TRAIN_PYTHON")
    if configured:
        return Path(configured)
    candidate = ROOT / ".venv-train" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else Path(sys.executable)


def _read_state() -> dict:
    if not STATE.is_file():
        return {"status": "idle"}
    try:
        return json.loads(STATE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {"status": "unknown"}


def _process_exists(pid: int) -> bool:
    if os.name == "nt":
        result = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}", "/FO", "CSV", "/NH"],
            capture_output=True,
            text=True,
        )
        return f'"{pid}"' in result.stdout
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def start_training(
    model,
    epochs,
    imgsz,
    batch,
    patience,
    fold,
    target,
    run_name=None,
    distill_model=None,
    distill_weight=6.0,
    optimizer="auto",
    lr0=0.01,
) -> str:
    state = _read_state()
    if state.get("status") == "running":
        if _process_exists(int(state["pid"])):
            return f"이미 학습 중입니다. PID {state['pid']}"
    python = training_python()
    if not python.is_file():
        return f"학습 환경이 없습니다: {python}. setup_training.ps1을 먼저 실행하세요."
    cmd = [str(python), str(ROOT / "train_student.py"), "--model", str(model or "yolo26n.pt"),
           "--epochs", str(int(epochs)), "--imgsz", str(int(imgsz)), "--batch", str(batch),
           "--patience", str(int(patience)), "--fold", str(fold), "--target-map50", str(float(target)), "--device", "auto"]
    if run_name:
        cmd.extend(["--run-name", str(run_name)])
    if distill_model:
        cmd.extend(["--distill-model", str(distill_model)])
    cmd.extend(
        [
            "--distill-weight",
            str(float(distill_weight)),
            "--optimizer",
            str(optimizer),
            "--lr0",
            str(float(lr0)),
        ]
    )
    LOG.parent.mkdir(parents=True, exist_ok=True)
    handle = LOG.open("w", encoding="utf-8")
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    environment = dict(os.environ)
    environment["PYTHONUTF8"] = "1"
    process = subprocess.Popen(
        cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
        creationflags=flags, env=environment,
    )
    handle.close()
    STATE.write_text(json.dumps({"status": "running", "pid": process.pid, "command": cmd}, ensure_ascii=False, indent=2), encoding="utf-8")
    return f"학습을 시작했습니다. PID {process.pid}. 새로고침 버튼으로 로그를 확인하세요."


def training_status() -> tuple[str, dict | None]:
    state = _read_state()
    log = LOG.read_text(encoding="utf-8", errors="replace") if LOG.is_file() else ""
    tail = "\n".join(log.splitlines()[-120:])
    summary_path = ROOT / "runs" / "student_cv" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
    status = state.get("status", "idle")
    if status == "running" and state.get("pid"):
        if not _process_exists(int(state["pid"])):
            status = "failed" if "Traceback (most recent call last)" in log and "SUMMARY " not in log else "finished"
            state["status"] = status
            STATE.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    report = f"### 상태: {status}\n\n```text\n{tail or '아직 로그가 없습니다.'}\n```"
    return report, summary
