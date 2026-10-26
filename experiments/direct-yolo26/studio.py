"""Local Gradio studio for labeling, dataset inspection, and turn validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from rummikub.annotations import (
    DATA_ROOT,
    dataset_summary,
    draw_boxes,
    normalize_box,
    save_yolo_sample,
)
from rummikub.classes import CLASS_NAMES, CLASS_TO_ID
from rummikub.detector import detect_tiles
from rummikub.rules import TurnValidation, validate_meld, validate_turn
from rummikub.smart_label import (
    local_yolo_auto_boxes,
    roboflow_auto_boxes,
    smart_box_from_point,
)


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = ROOT / "runs" / "detect" / "rummikub_yolo26n" / "weights" / "best.pt"
TOOL_SMART = "스마트 박스 (한 번 클릭)"
TOOL_MANUAL = "수동 박스 (두 모서리)"
TOOL_RELABEL = "클릭한 박스 클래스 변경"
TOOL_DELETE = "클릭한 박스 삭제"
APP_CSS = """
.hero { text-align: center; margin-bottom: 12px; }
.status { min-height: 72px; }
"""


def reset_annotation(image: np.ndarray | None):
    if image is None:
        return None, [], None, None, "이미지를 업로드하거나 웹캠으로 촬영하세요.", None
    original = np.asarray(image).copy()
    return original, [], None, original, "클릭 도구를 선택해 라벨을 추가하거나 수정하세요.", None


def dataset_image_choices(split: str) -> list[str]:
    directory = DATA_ROOT / "images" / split
    if not directory.is_dir():
        return []
    return sorted(path.name for path in directory.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp"})


def refresh_dataset_choices(split: str):
    choices = dataset_image_choices(split)
    return gr.update(choices=choices, value=choices[0] if choices else None)


def load_existing_sample(split: str, filename: str | None):
    if not filename:
        return None, [], None, None, "불러올 데이터셋 이미지를 선택하세요.", None
    image_path = DATA_ROOT / "images" / split / Path(filename).name
    label_path = DATA_ROOT / "labels" / split / f"{image_path.stem}.txt"
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None, [], None, None, f"이미지를 읽지 못했습니다: {image_path}", None
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    boxes: list[dict[str, Any]] = []
    if label_path.is_file():
        for line in label_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            class_id, center_x, center_y, box_width, box_height = map(float, line.split())
            boxes.append(
                {
                    "class_id": int(class_id),
                    "x1": (center_x - box_width / 2) * width,
                    "y1": (center_y - box_height / 2) * height,
                    "x2": (center_x + box_width / 2) * width,
                    "y2": (center_y + box_height / 2) * height,
                    "source": "dataset",
                }
            )
    sample = {"image": str(image_path), "label": str(label_path), "split": split}
    return image, boxes, None, draw_boxes(image, boxes), f"✅ 기존 라벨 {len(boxes)}개를 불러왔습니다.", sample


def add_corner(
    original: np.ndarray | None,
    boxes: list[dict[str, Any]] | None,
    pending: tuple[int, int] | None,
    class_name: str,
    evt: gr.SelectData,
):
    if original is None:
        return None, boxes or [], pending, "먼저 이미지를 입력하세요."
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return draw_boxes(original, boxes or [], pending), boxes or [], pending, "좌표를 읽지 못했습니다. 다시 클릭하세요."
    point = (int(index[0]), int(index[1]))
    boxes = [dict(box) for box in (boxes or [])]
    if pending is None:
        return draw_boxes(original, boxes, point), boxes, point, f"첫 모서리 {point}. 반대쪽 모서리를 클릭하세요."
    x1, x2 = sorted((pending[0], point[0]))
    y1, y2 = sorted((pending[1], point[1]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return draw_boxes(original, boxes), boxes, None, "박스가 너무 작아 취소했습니다."
    boxes.append(
        {"class_id": CLASS_TO_ID[class_name], "x1": x1, "y1": y1, "x2": x2, "y2": y2}
    )
    return draw_boxes(original, boxes), boxes, None, f"{class_name} 박스를 추가했습니다. 현재 {len(boxes)}개."


def _box_at_point(boxes: list[dict[str, Any]], point: tuple[int, int]) -> int | None:
    x, y = point
    matches = [
        (index, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))
        for index, box in enumerate(boxes)
        if box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]
    ]
    return min(matches, key=lambda item: item[1])[0] if matches else None


def handle_canvas_click(
    original: np.ndarray | None,
    boxes: list[dict[str, Any]] | None,
    pending: tuple[int, int] | None,
    class_name: str,
    tool_mode: str,
    evt: gr.SelectData,
):
    if original is None:
        return None, boxes or [], pending, "먼저 이미지를 입력하세요."
    index = evt.index
    if not isinstance(index, (tuple, list)) or len(index) < 2:
        return draw_boxes(original, boxes or [], pending), boxes or [], pending, "좌표를 읽지 못했습니다."
    point = (int(index[0]), int(index[1]))
    boxes = [dict(box) for box in (boxes or [])]

    if tool_mode == TOOL_SMART:
        try:
            boxes.append(smart_box_from_point(original, point, CLASS_TO_ID[class_name], boxes))
        except Exception as exc:
            return draw_boxes(original, boxes), boxes, None, f"❌ 스마트 박스 실패: {exc}"
        return draw_boxes(original, boxes), boxes, None, f"✅ MobileSAM이 {class_name} 경계를 생성했습니다."

    selected = _box_at_point(boxes, point)
    if tool_mode == TOOL_DELETE:
        if selected is None:
            return draw_boxes(original, boxes), boxes, None, "클릭한 위치에 삭제할 박스가 없습니다."
        removed = boxes.pop(selected)
        return draw_boxes(original, boxes), boxes, None, f"{CLASS_NAMES[int(removed['class_id'])]} 박스를 삭제했습니다."
    if tool_mode == TOOL_RELABEL:
        if selected is None:
            return draw_boxes(original, boxes), boxes, None, "클릭한 위치에 변경할 박스가 없습니다."
        boxes[selected]["class_id"] = CLASS_TO_ID[class_name]
        boxes[selected]["source"] = "human_review"
        return draw_boxes(original, boxes), boxes, None, f"선택한 박스를 {class_name}(으)로 변경했습니다."

    return add_corner(original, boxes, pending, class_name, evt)


def auto_label_image(original, source_name, confidence, local_weights):
    if original is None:
        return None, [], None, "❌ 먼저 이미지를 입력하세요."
    try:
        if source_name == "Roboflow 기존 Rummikub 모델":
            boxes = roboflow_auto_boxes(original, confidence)
            source = "Roboflow"
        else:
            boxes = local_yolo_auto_boxes(original, local_weights, confidence)
            source = "로컬 YOLO26n"
    except Exception as exc:
        return original, [], None, f"❌ 자동 라벨링 실패: {exc}"
    return (
        draw_boxes(original, boxes),
        boxes,
        None,
        f"✅ {source} 초안 {len(boxes)}개를 만들었습니다. 클래스 변경/삭제/스마트 박스로 검수하세요.",
    )


def undo_box(original, boxes, pending):
    boxes = [dict(box) for box in (boxes or [])]
    if pending is not None:
        pending = None
        message = "선택 중이던 첫 모서리를 취소했습니다."
    elif boxes:
        removed = boxes.pop()
        message = f'{CLASS_NAMES[int(removed["class_id"])]} 박스를 삭제했습니다.'
    else:
        message = "삭제할 박스가 없습니다."
    return draw_boxes(original, boxes, pending) if original is not None else None, boxes, pending, message


def clear_boxes(original):
    return original, [], None, "모든 박스를 지웠습니다."


def save_annotation(original, boxes, split, allow_empty, loaded_sample, overwrite_loaded):
    boxes = boxes or []
    if original is None:
        return "❌ 저장할 이미지가 없습니다.", dataset_summary()
    if not boxes and not allow_empty:
        return "❌ 박스가 없습니다. 배경 이미지라면 '타일 없는 이미지'를 체크하세요.", dataset_summary()
    try:
        if loaded_sample and overwrite_loaded:
            image_path = Path(loaded_sample["image"])
            label_path = Path(loaded_sample["label"])
            height, width = np.asarray(original).shape[:2]
            lines = [normalize_box(box, width, height) for box in boxes]
            label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            action = "검수 라벨을 덮어썼습니다"
        else:
            image_path, label_path = save_yolo_sample(original, boxes, split)
            action = f"`{split}`에 새 샘플로 저장했습니다"
    except Exception as exc:
        return f"❌ 저장 실패: {exc}", dataset_summary()
    return (
        f"✅ {action}: `{image_path.name}` / `{label_path.name}`",
        dataset_summary(),
    )


def set_baseline(image):
    if image is None:
        return None, None, "❌ 기준 이미지를 먼저 촬영하세요."
    image = np.asarray(image).copy()
    return image, image, "✅ 이전 턴 기준 이미지를 저장했습니다."


def _turn_payload(validation: TurnValidation) -> dict[str, Any]:
    return {
        "valid": validation.valid,
        "reasons": validation.reasons,
        "warnings": validation.warnings,
        "added_tiles": dict(validation.added),
        "initial_score": validation.initial_score,
        "melds": [
            {
                "tiles": [tile.class_name for tile in meld],
                "validation": validate_meld(meld).__dict__,
            }
            for meld in validation.melds
        ],
    }


def check_and_commit_turn(
    current_image,
    previous_image,
    weights,
    initial_meld_done,
    confidence,
    image_size,
    gap_factor,
):
    if previous_image is None:
        return None, None, "❌ 먼저 '기준 턴 저장'을 누르세요.", None, None, None
    if current_image is None:
        return None, None, "❌ 현재 턴 이미지를 촬영하세요.", previous_image, previous_image, None
    try:
        previous_tiles, _ = detect_tiles(previous_image, weights, confidence, int(image_size))
        current_tiles, annotated = detect_tiles(current_image, weights, confidence, int(image_size))
        validation = validate_turn(previous_tiles, current_tiles, initial_meld_done, gap_factor)
    except Exception as exc:
        return None, previous_image, f"❌ 판정 실패: {exc}", previous_image, previous_image, {"error": str(exc)}

    payload = _turn_payload(validation)
    if validation.valid:
        warning_text = "\n".join(f"- ⚠️ {warning}" for warning in validation.warnings)
        report = (
            f"## ✅ 정상 턴\n\n새 타일: `{dict(validation.added)}`"
            + (f"  \n최초 등록 자동 합계: **{validation.initial_score}점**" if not initial_meld_done else "")
            + (f"\n\n{warning_text}" if warning_text else "")
        )
        committed = np.asarray(current_image).copy()
        return annotated, None, report, committed, committed, payload

    reasons = "\n".join(f"- {reason}" for reason in validation.reasons)
    report = f"## ❌ 룰 위반 - 이전 턴으로 복원\n\n{reasons}"
    return annotated, previous_image, report, previous_image, previous_image, payload


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Rummikub YOLO26 Studio") as demo:
        gr.Markdown(
            "# Rummikub YOLO26 Studio\n"
            "로컬 데이터 라벨링, YOLO26n 학습 준비, 턴별 룰 검증을 한 화면에서 수행합니다.",
            elem_classes="hero",
        )

        with gr.Tab("1. 데이터 라벨링"):
            original_state = gr.State(None)
            boxes_state = gr.State([])
            pending_state = gr.State(None)
            loaded_sample_state = gr.State(None)
            with gr.Accordion("기존 데이터셋 라벨 검수", open=True):
                with gr.Row():
                    browser_split = gr.Radio(["train", "val", "test"], value="train", label="불러올 분할")
                    browser_image = gr.Dropdown(
                        choices=dataset_image_choices("train"),
                        label="이미지 검색/선택",
                        filterable=True,
                    )
                    load_existing = gr.Button("기존 이미지와 라벨 불러오기")
            with gr.Row():
                source = gr.Image(
                    sources=["upload", "webcam"],
                    type="numpy",
                    label="원본 이미지 / 웹캠 촬영",
                    height=520,
                )
                preview = gr.Image(
                    type="numpy",
                    label="라벨링 화면 - 타일의 두 모서리를 클릭",
                    interactive=True,
                    height=520,
                )
            with gr.Row():
                class_name = gr.Dropdown(CLASS_NAMES, value="black_1", label="타일 클래스", filterable=True)
                tool_mode = gr.Radio(
                    [TOOL_SMART, TOOL_MANUAL, TOOL_RELABEL, TOOL_DELETE],
                    value=TOOL_SMART,
                    label="클릭 도구",
                )
                split = gr.Radio(["train", "val", "test"], value="train", label="데이터 분할")
                allow_empty = gr.Checkbox(False, label="타일 없는 배경 이미지")
                overwrite_loaded = gr.Checkbox(True, label="불러온 라벨 파일 덮어쓰기")
            with gr.Accordion("자동 라벨링", open=True):
                with gr.Row():
                    auto_source = gr.Radio(
                        ["Roboflow 기존 Rummikub 모델", "로컬 YOLO26n best.pt"],
                        value="Roboflow 기존 Rummikub 모델",
                        label="초안 생성 모델",
                    )
                    auto_confidence = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="자동 라벨 신뢰도")
                    local_label_weights = gr.Textbox(value=str(DEFAULT_WEIGHTS), label="로컬 가중치")
                auto_label = gr.Button("자동 라벨 초안 생성", variant="secondary")
                gr.Markdown(
                    "Roboflow를 선택하면 이미지가 API 서버로 전송됩니다. 초안은 반드시 사람이 확인한 뒤 저장하세요."
                )
            with gr.Row():
                undo = gr.Button("마지막 박스/클릭 취소")
                clear = gr.Button("박스 전체 삭제")
                save = gr.Button("YOLO 라벨 저장", variant="primary")
                refresh = gr.Button("현황 새로고침")
            label_status = gr.Markdown("이미지를 입력하세요.", elem_classes="status")
            summary = gr.Markdown(dataset_summary())

            source.change(
                reset_annotation,
                inputs=source,
                outputs=[original_state, boxes_state, pending_state, preview, label_status, loaded_sample_state],
            )
            browser_split.change(refresh_dataset_choices, inputs=browser_split, outputs=browser_image)
            load_existing.click(
                load_existing_sample,
                inputs=[browser_split, browser_image],
                outputs=[original_state, boxes_state, pending_state, preview, label_status, loaded_sample_state],
            )
            preview.select(
                handle_canvas_click,
                inputs=[original_state, boxes_state, pending_state, class_name, tool_mode],
                outputs=[preview, boxes_state, pending_state, label_status],
            )
            auto_label.click(
                auto_label_image,
                inputs=[original_state, auto_source, auto_confidence, local_label_weights],
                outputs=[preview, boxes_state, pending_state, label_status],
            )
            undo.click(
                undo_box,
                inputs=[original_state, boxes_state, pending_state],
                outputs=[preview, boxes_state, pending_state, label_status],
            )
            clear.click(
                clear_boxes,
                inputs=original_state,
                outputs=[preview, boxes_state, pending_state, label_status],
            )
            save.click(
                save_annotation,
                inputs=[original_state, boxes_state, split, allow_empty, loaded_sample_state, overwrite_loaded],
                outputs=[label_status, summary],
            )
            refresh.click(dataset_summary, outputs=summary)

        with gr.Tab("2. 턴 룰 검증"):
            previous_state = gr.State(None)
            gr.Markdown(
                "고정된 상단 카메라를 사용하세요. 먼저 턴 시작 전 테이블을 기준으로 저장하고, "
                "플레이 후 현재 턴을 촬영해 판정합니다. 정상 턴만 새 기준으로 자동 확정됩니다."
            )
            with gr.Row():
                baseline_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="턴 시작 전 기준 이미지")
                previous_display = gr.Image(type="numpy", label="현재 확정된 이전 턴", interactive=False)
            baseline_button = gr.Button("기준 턴 저장", variant="secondary")
            baseline_status = gr.Markdown()
            baseline_button.click(
                set_baseline,
                inputs=baseline_input,
                outputs=[previous_state, previous_display, baseline_status],
            )
            with gr.Row():
                current_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="플레이 후 현재 턴")
                detected_output = gr.Image(type="numpy", label="YOLO26n 탐지 결과", interactive=False)
                rollback_output = gr.Image(type="numpy", label="위반 시 복원할 이전 턴 이미지", interactive=False)
            with gr.Row():
                weights = gr.Textbox(value=str(DEFAULT_WEIGHTS), label="학습 가중치(best.pt)")
                initial_done = gr.Checkbox(False, label="현재 플레이어가 최초 등록을 완료함")
            with gr.Accordion("판정 설정", open=False):
                confidence = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="탐지 신뢰도")
                image_size = gr.Slider(640, 1280, value=960, step=32, label="YOLO 입력 크기")
                gap_factor = gr.Slider(0.3, 1.5, value=0.75, step=0.05, label="세트 사이 간격 계수")
            check_button = gr.Button("현재 턴 판정 및 정상 시 확정", variant="primary")
            rule_report = gr.Markdown(elem_classes="status")
            raw_result = gr.JSON(label="상세 판정 데이터")
            check_button.click(
                check_and_commit_turn,
                inputs=[current_input, previous_state, weights, initial_done, confidence, image_size, gap_factor],
                outputs=[detected_output, rollback_output, rule_report, previous_state, previous_display, raw_result],
            )

        with gr.Tab("3. 학습 안내"):
            gr.Markdown(
                "## 학습 전 검사\n"
                "```powershell\n.\\.venv\\Scripts\\python.exe validate_dataset.py\n```\n"
                "## YOLO26 Nano 학습\n"
                "```powershell\n.\\.venv\\Scripts\\python.exe train.py --epochs 100 --imgsz 960\n```\n"
                "학습 결과는 `runs/detect/rummikub_yolo26n/weights/best.pt`에 저장됩니다."
            )
    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rummikub YOLO26 local studio")
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
        css=APP_CSS,
    )
