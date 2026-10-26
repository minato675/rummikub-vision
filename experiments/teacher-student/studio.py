"""Gradio studio for local Roboflow labeling, testing, and layout validation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import gradio as gr
import numpy as np

from rummikub.classes import CLASS_NAMES, CLASS_TO_ID
from rummikub.dataset import (
    IMAGE_DIR,
    LABEL_DIR,
    export_roboflow_zip,
    image_choices,
    load_boxes,
    overwrite_labels,
    save_sample,
    summary,
)
from rummikub.evaluation import evaluate_dataset, prediction_boxes
from rummikub.rules import Tile, validate_layout
from rummikub.visuals import draw_box_label, draw_invalid_highlights
from rummikub_inference import build_client, draw_predictions, infer, load_settings
from rummikub.quality import audit_dataset
from training_control import start_training, training_status


TOOL_SMART = "모델 박스 (한 번 클릭)"
TOOL_MANUAL = "수동 박스 (두 모서리)"
TOOL_RELABEL = "클릭한 박스 클래스 변경"
TOOL_DELETE = "클릭한 박스 삭제"


def draw_label_boxes(image: np.ndarray, boxes: list[dict[str, Any]], pending=None) -> np.ndarray:
    canvas = np.asarray(image).copy()
    for box in boxes:
        x1, y1, x2, y2 = (int(box[key]) for key in ("x1", "y1", "x2", "y2"))
        label = CLASS_NAMES[int(box["class_id"])]
        confidence = box.get("confidence")
        display = f"{label}  {float(confidence):.0%}" if confidence is not None else label
        draw_box_label(canvas, (x1, y1, x2, y2), display, label, channel_order="RGB")
    if pending is not None:
        cv2.drawMarker(canvas, pending, (255, 200, 0), cv2.MARKER_CROSS, 24, 3)
    return canvas


def infer_rgb(image: np.ndarray, confidence: float):
    client, settings = build_client(confidence)
    bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
    return infer(client, settings.model_id, bgr), settings


def reset_annotation(image):
    if image is None:
        return None, [], None, None, "이미지를 업로드하거나 웹캠으로 촬영하세요.", None
    original = np.asarray(image).copy()
    return original, [], None, original, "자동 라벨 또는 편집 도구를 선택하세요.", None


def refresh_dataset():
    choices = image_choices()
    return gr.update(choices=choices, value=choices[0] if choices else None), summary()


def load_existing(filename: str | None):
    if not filename:
        return None, [], None, None, "불러올 이미지를 선택하세요.", None
    image_path = IMAGE_DIR / Path(filename).name
    label_path = LABEL_DIR / f"{image_path.stem}.txt"
    bgr = cv2.imread(str(image_path))
    if bgr is None:
        return None, [], None, None, f"이미지를 읽을 수 없습니다: {image_path}", None
    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    height, width = image.shape[:2]
    boxes = load_boxes(label_path, width, height)
    loaded = {"image": str(image_path), "label": str(label_path)}
    return image, boxes, None, draw_label_boxes(image, boxes), f"기존 라벨 {len(boxes)}개를 불러왔습니다.", loaded


def auto_label(image, confidence):
    if image is None:
        return None, [], None, "먼저 이미지를 입력하세요."
    try:
        result, _ = infer_rgb(image, confidence)
        boxes = prediction_boxes(result)
    except Exception as exc:
        return image, [], None, f"자동 라벨 실패: {exc}"
    return draw_label_boxes(image, boxes), boxes, None, f"로컬 Roboflow 모델이 {len(boxes)}개의 초안을 만들었습니다. 반드시 검수하세요."


def _box_at_point(boxes, point):
    x, y = point
    matches = [
        (index, (box["x2"] - box["x1"]) * (box["y2"] - box["y1"]))
        for index, box in enumerate(boxes)
        if box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]
    ]
    return min(matches, key=lambda item: item[1])[0] if matches else None


def handle_click(image, boxes, pending, class_name, tool, confidence, evt: gr.SelectData):
    if image is None:
        return None, boxes or [], pending, "먼저 이미지를 입력하세요."
    if not isinstance(evt.index, (tuple, list)) or len(evt.index) < 2:
        return draw_label_boxes(image, boxes or [], pending), boxes or [], pending, "클릭 좌표를 읽지 못했습니다."
    point = (int(evt.index[0]), int(evt.index[1]))
    boxes = [dict(box) for box in (boxes or [])]

    if tool == TOOL_SMART:
        try:
            result, _ = infer_rgb(image, confidence)
            candidates = prediction_boxes(result)
            selected = _box_at_point(candidates, point)
            if selected is None:
                return draw_label_boxes(image, boxes), boxes, None, "클릭 위치에서 모델 박스를 찾지 못했습니다. 수동 박스를 사용하세요."
            candidate = dict(candidates[selected])
            candidate["class_id"] = CLASS_TO_ID[class_name]
            candidate["source"] = "model_click_human_class"
            boxes.append(candidate)
            return draw_label_boxes(image, boxes), boxes, None, f"모델 경계로 {class_name} 박스를 추가했습니다."
        except Exception as exc:
            return draw_label_boxes(image, boxes), boxes, None, f"모델 박스 실패: {exc}"

    selected = _box_at_point(boxes, point)
    if tool == TOOL_DELETE:
        if selected is None:
            return draw_label_boxes(image, boxes), boxes, None, "클릭 위치에 삭제할 박스가 없습니다."
        removed = boxes.pop(selected)
        return draw_label_boxes(image, boxes), boxes, None, f"{CLASS_NAMES[int(removed['class_id'])]} 박스를 삭제했습니다."
    if tool == TOOL_RELABEL:
        if selected is None:
            return draw_label_boxes(image, boxes), boxes, None, "클릭 위치에 변경할 박스가 없습니다."
        boxes[selected]["class_id"] = CLASS_TO_ID[class_name]
        boxes[selected]["source"] = "human_review"
        return draw_label_boxes(image, boxes), boxes, None, f"클래스를 {class_name}으로 변경했습니다."

    if pending is None:
        return draw_label_boxes(image, boxes, point), boxes, point, "첫 모서리를 선택했습니다. 반대쪽 모서리를 클릭하세요."
    x1, x2 = sorted((pending[0], point[0]))
    y1, y2 = sorted((pending[1], point[1]))
    if x2 - x1 < 2 or y2 - y1 < 2:
        return draw_label_boxes(image, boxes), boxes, None, "박스가 너무 작아 취소했습니다."
    boxes.append({"class_id": CLASS_TO_ID[class_name], "x1": x1, "y1": y1, "x2": x2, "y2": y2, "source": "manual"})
    return draw_label_boxes(image, boxes), boxes, None, f"{class_name} 박스를 추가했습니다."


def undo(image, boxes, pending):
    boxes = [dict(box) for box in (boxes or [])]
    if pending is not None:
        return draw_label_boxes(image, boxes), boxes, None, "첫 모서리 선택을 취소했습니다."
    if boxes:
        removed = boxes.pop()
        return draw_label_boxes(image, boxes), boxes, None, f"{CLASS_NAMES[int(removed['class_id'])]} 박스를 취소했습니다."
    return image, boxes, None, "취소할 박스가 없습니다."


def clear(image):
    return image, [], None, "모든 박스를 지웠습니다."


def save_annotation(image, boxes, allow_empty, loaded, overwrite):
    boxes = boxes or []
    if image is None:
        return "저장할 이미지가 없습니다.", summary()
    if not boxes and not allow_empty:
        return "박스가 없습니다. 배경 이미지라면 '타일 없는 이미지'를 체크하세요.", summary()
    try:
        if loaded and overwrite:
            label_path = Path(loaded["label"])
            overwrite_labels(label_path, np.asarray(image), boxes)
            message = f"기존 라벨을 덮어썼습니다: {label_path.name}"
        else:
            image_path, label_path = save_sample(np.asarray(image), boxes)
            message = f"새 샘플을 저장했습니다: {image_path.name} / {label_path.name}"
    except Exception as exc:
        return f"저장 실패: {exc}", summary()
    return message, summary()


def test_model(image, confidence):
    if image is None:
        return None, "이미지를 입력하세요.", None
    try:
        result, settings = infer_rgb(image, confidence)
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        annotated = cv2.cvtColor(draw_predictions(bgr, result), cv2.COLOR_BGR2RGB)
        count = len(result.get("predictions", []))
        return annotated, f"**{settings.model_id}** — 타일 **{count}개** 탐지", result
    except Exception as exc:
        return image, f"테스트 실패: {exc}", {"error": str(exc)}


def run_dataset_evaluation(confidence, max_images):
    try:
        result = evaluate_dataset(confidence, int(max_images))
    except Exception as exc:
        return f"평가 실패: {exc}", {"error": str(exc)}
    report = (
        f"### IoU@0.5 간이 평가\n\n"
        f"- 이미지: **{result['images']}장**\n"
        f"- Precision: **{result['precision']:.3f}**\n"
        f"- Recall: **{result['recall']:.3f}**\n"
        f"- F1: **{result['f1']:.3f}**\n\n"
        "COCO mAP이 아닌 저장 라벨과의 간이 점검 결과입니다."
    )
    return report, result


def check_layout(image, confidence, gap_factor):
    if image is None:
        return None, "현재 테이블 이미지를 입력하세요.", None
    try:
        result, settings = infer_rgb(image, confidence)
        boxes = prediction_boxes(result)
        tiles = [
            Tile(
                CLASS_NAMES[int(box["class_id"])],
                float(box.get("confidence", 1)),
                float(box["x1"]), float(box["y1"]), float(box["x2"]), float(box["y2"]),
            )
            for box in boxes
        ]
        validation = validate_layout(tiles, float(gap_factor))
        bgr = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)
        annotated = cv2.cvtColor(draw_predictions(bgr, result), cv2.COLOR_BGR2RGB)
        violation_sets = {
            tile_index: meld["index"]
            for meld in validation["melds"]
            for tile_index in meld["offending_tile_indices"]
        }
        annotated = draw_invalid_highlights(annotated, boxes, violation_sets, channel_order="RGB")
        validation["highlighted_tile_count"] = len(violation_sets)
    except Exception as exc:
        return image, f"배치 판정 실패: {exc}", {"error": str(exc)}

    lines = ["## ✅ 현재 배치 정상" if validation["valid"] else "## ❌ 현재 배치 확인 필요"]
    lines.append(f"모델 `{settings.model_id}`이 탐지한 타일: **{validation['tile_count']}개**")
    if validation["highlighted_tile_count"]:
        lines.append(f"🚨 규칙 위반 의심 타일 **{validation['highlighted_tile_count']}개**를 빨간 X로 강조했습니다.")
    for meld in validation["melds"]:
        icon = "✅" if meld["valid"] else "❌"
        kind = {"group": "그룹", "run": "연속"}.get(meld["kind"], "위반")
        lines.append(f"- {icon} 세트 {meld['index']}: `{', '.join(meld['tiles'])}` — {kind}; {meld['reason']}")
    lines.append(f"\n⚠️ {validation['warning']}")
    return annotated, "\n".join(lines), validation


def make_export():
    try:
        path = export_roboflow_zip()
        return str(path), f"Roboflow 업로드용 ZIP을 만들었습니다: `{path.name}`"
    except Exception as exc:
        return None, f"내보내기 실패: {exc}"


def audit_for_studio():
    try:
        result = audit_dataset()
    except Exception as exc:
        return f"### 데이터 검사 실패\n\n{exc}", {"error": str(exc)}
    icon = "✅ 학습 준비 완료" if result["ready"] else "⚠️ 보완 필요"
    report = (
        f"### {icon}\n\n"
        f"- 이미지: **{result['images']}장**\n"
        f"- 박스: **{result['boxes']}개**\n"
        f"- 누락 클래스: **{len(result['missing_classes'])}개**\n"
        f"- 20개 미만 희소 클래스: **{len(result['rare_classes_under_20'])}개**\n\n"
        "학습 준비 완료는 최소 100장, 53개 클래스 포함, 누락/잘못된 라벨 없음 조건입니다."
    )
    return report, result


def build_app() -> gr.Blocks:
    settings = load_settings(require_api_key=False)
    with gr.Blocks(title="Rummikub Local Roboflow Studio") as demo:
        gr.Markdown(
            "# Rummikub Local Roboflow Studio\n"
            f"로컬 서버 `{settings.server_url}`의 모델 `{settings.model_id}`을 사용합니다."
        )

        with gr.Tab("1. 추가 라벨링"):
            original_state = gr.State(None)
            boxes_state = gr.State([])
            pending_state = gr.State(None)
            loaded_state = gr.State(None)
            with gr.Accordion("저장된 추가 라벨 불러오기", open=False):
                with gr.Row():
                    browser = gr.Dropdown(image_choices(), label="저장 이미지", filterable=True)
                    load_button = gr.Button("이미지와 라벨 불러오기")
                    refresh_button = gr.Button("목록 새로고침")
            with gr.Row():
                source = gr.Image(sources=["upload", "webcam"], type="numpy", label="원본 이미지 / 웹캠", height=500)
                preview = gr.Image(type="numpy", label="라벨 편집 화면", interactive=True, height=500)
            with gr.Row():
                class_name = gr.Dropdown(CLASS_NAMES, value="black_1", label="타일 클래스", filterable=True)
                tool = gr.Radio([TOOL_SMART, TOOL_MANUAL, TOOL_RELABEL, TOOL_DELETE], value=TOOL_SMART, label="클릭 도구")
                label_confidence = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="모델 신뢰도")
            auto_button = gr.Button("로컬 모델로 자동 라벨 초안 생성", variant="secondary")
            with gr.Row():
                undo_button = gr.Button("마지막 작업 취소")
                clear_button = gr.Button("박스 전체 삭제")
                allow_empty = gr.Checkbox(False, label="타일 없는 이미지")
                overwrite = gr.Checkbox(True, label="불러온 라벨 덮어쓰기")
                save_button = gr.Button("검수 완료 라벨 저장", variant="primary")
            label_status = gr.Markdown("이미지를 입력하세요.")
            dataset_status = gr.Markdown(summary())

            source.change(reset_annotation, inputs=source, outputs=[original_state, boxes_state, pending_state, preview, label_status, loaded_state])
            load_button.click(load_existing, inputs=browser, outputs=[original_state, boxes_state, pending_state, preview, label_status, loaded_state])
            refresh_button.click(refresh_dataset, outputs=[browser, dataset_status])
            auto_button.click(auto_label, inputs=[original_state, label_confidence], outputs=[preview, boxes_state, pending_state, label_status])
            preview.select(handle_click, inputs=[original_state, boxes_state, pending_state, class_name, tool, label_confidence], outputs=[preview, boxes_state, pending_state, label_status])
            undo_button.click(undo, inputs=[original_state, boxes_state, pending_state], outputs=[preview, boxes_state, pending_state, label_status])
            clear_button.click(clear, inputs=original_state, outputs=[preview, boxes_state, pending_state, label_status])
            save_button.click(save_annotation, inputs=[original_state, boxes_state, allow_empty, loaded_state, overwrite], outputs=[label_status, dataset_status])

        with gr.Tab("2. 모델 테스트"):
            gr.Markdown("단일 이미지 육안 테스트와 사람이 검수한 저장 라벨에 대한 IoU@0.5 간이 평가를 제공합니다.")
            with gr.Row():
                test_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="테스트 이미지")
                test_output = gr.Image(type="numpy", label="탐지 결과")
            test_confidence = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="신뢰도")
            test_button = gr.Button("단일 이미지 테스트", variant="primary")
            test_report = gr.Markdown()
            test_json = gr.JSON(label="원본 탐지 응답")
            test_button.click(test_model, inputs=[test_input, test_confidence], outputs=[test_output, test_report, test_json])
            with gr.Accordion("저장 데이터셋 간이 평가", open=True):
                max_images = gr.Slider(1, 500, value=100, step=1, label="최대 평가 이미지")
                evaluate_button = gr.Button("저장 라벨과 모델 비교")
                evaluation_report = gr.Markdown()
                evaluation_json = gr.JSON(label="평가 상세")
                evaluate_button.click(run_dataset_evaluation, inputs=[test_confidence, max_images], outputs=[evaluation_report, evaluation_json])

        with gr.Tab("3. 현재 배치 판정"):
            gr.Markdown("고정된 상단 카메라로 현재 테이블 전체를 촬영하세요. 타일 위치를 세트로 군집화한 뒤 그룹·연속 규칙을 검사합니다.")
            with gr.Row():
                layout_input = gr.Image(sources=["upload", "webcam"], type="numpy", label="현재 테이블")
                layout_output = gr.Image(type="numpy", label="탐지된 타일")
            with gr.Row():
                layout_confidence = gr.Slider(0.05, 0.95, value=0.35, step=0.05, label="신뢰도")
                gap_factor = gr.Slider(0.3, 1.5, value=0.75, step=0.05, label="세트 사이 간격 계수")
            layout_button = gr.Button("현재 배치 검사", variant="primary")
            layout_report = gr.Markdown()
            layout_json = gr.JSON(label="배치 판정 상세")
            layout_button.click(check_layout, inputs=[layout_input, layout_confidence, gap_factor], outputs=[layout_output, layout_report, layout_json])

        with gr.Tab("4. 추가 학습 준비"):
            gr.Markdown(
                "## 추가 학습 방식\n\n"
                "로컬 Docker 캐시는 **추론 전용**이므로 공개 모델 가중치를 이 화면에서 직접 학습하지 않습니다. "
                "검수 데이터를 본인의 Roboflow 프로젝트에 올리고 새 Dataset Version과 모델을 학습해야 합니다.\n\n"
                "1. Universe의 `rummikub-p8akb`를 내 Workspace로 Fork\n"
                "2. 아래 ZIP을 내 프로젝트에 업로드하고 클래스 매핑 확인\n"
                "3. Dataset Version 생성 후 Roboflow Train 실행\n"
                "4. 가능하면 기존 모델/공개 모델을 checkpoint 또는 transfer-learning 시작점으로 선택\n"
                "5. 새 모델 ID를 `.env`의 `ROBOFLOW_MODEL_ID`에 입력\n"
                "6. 로컬 서버와 Studio를 다시 시작해 새 버전 테스트\n\n"
                "원본 공개 모델은 수정되지 않으며, 개선 모델은 **내 Workspace의 새 모델 버전**으로 생성됩니다."
            )
            export_button = gr.Button("검수 데이터 Roboflow ZIP 만들기", variant="primary")
            export_file = gr.File(label="업로드용 ZIP")
            export_status = gr.Markdown()
            export_button.click(make_export, outputs=[export_file, export_status])

        with gr.Tab("5. 로컬 YOLO26n 학습"):
            gr.Markdown(
                "## Teacher → Student 로컬 학습\n\n"
                "Roboflow 모델이 만든 자동 라벨을 사람이 검수한 뒤 YOLO26n을 학습합니다. "
                "5-fold는 매번 80% 학습/20% 검증이며 동일 원본의 증강 이미지는 같은 fold에 묶어 데이터 누수를 막습니다. "
                "기본 합격 기준은 **mAP50 0.90 이상**입니다. 수치는 보장값이 아니라 검증 결과로 판정됩니다."
            )
            with gr.Row():
                audit_button = gr.Button("데이터 품질 검사", variant="secondary")
                refresh_training = gr.Button("학습 상태/로그 새로고침")
            audit_report = gr.Markdown()
            audit_json = gr.JSON(label="데이터 품질 상세")
            audit_button.click(audit_for_studio, outputs=[audit_report, audit_json])
            with gr.Row():
                train_model = gr.Textbox(value="yolo26n.pt", label="시작 가중치", info="다음 반복부터 이전 best.pt를 지정하면 이어서 개선합니다.")
                train_fold = gr.Dropdown(["1", "2", "3", "4", "5", "all"], value="1", label="Fold", info="빠른 실험은 1, 최종 평가는 all")
                target_map = gr.Slider(0.50, 0.99, value=0.90, step=0.01, label="목표 mAP50")
            with gr.Row():
                epochs = gr.Number(value=150, precision=0, label="최대 Epoch")
                imgsz = gr.Dropdown([640, 768, 960, 1280], value=960, label="이미지 크기")
                batch = gr.Number(value=-1, label="Batch (-1 자동)")
                patience = gr.Number(value=20, precision=0, label="조기 종료 patience")
            train_button = gr.Button("로컬 GPU 학습 시작", variant="primary")
            train_report = gr.Markdown("아직 학습을 시작하지 않았습니다.")
            train_summary = gr.JSON(label="교차검증 결과 / 90% 통과 여부")
            train_button.click(
                start_training,
                inputs=[train_model, epochs, imgsz, batch, patience, train_fold, target_map],
                outputs=train_report,
            )
            refresh_training.click(training_status, outputs=[train_report, train_summary])

    return demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-name", default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--no-browser", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    build_app().queue(default_concurrency_limit=2).launch(
        server_name=args.server_name,
        server_port=args.server_port,
        inbrowser=not args.no_browser,
    )
