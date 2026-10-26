"""Real-time webcam detection using the hosted Rummikub model."""

from __future__ import annotations

import argparse
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from dotenv import load_dotenv

from app import DEFAULT_MODEL_ID, create_client


@dataclass
class InferenceResult:
    predictions: list[dict[str, Any]]
    width: int
    height: int
    latency_ms: float


class AsyncDetector:
    """Keep camera display responsive while one hosted inference runs at a time."""

    def __init__(self, model_id: str, inference_width: int) -> None:
        self.client = create_client()
        self.model_id = model_id
        self.inference_width = inference_width
        self.frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self.latest: InferenceResult | None = None
        self.error: str | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def submit(self, frame: np.ndarray) -> bool:
        if self.frames.full():
            return False
        self.frames.put_nowait(frame.copy())
        return True

    def snapshot(self) -> tuple[InferenceResult | None, str | None]:
        with self._lock:
            return self.latest, self.error

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2)

    def clear(self) -> None:
        """Discard results from the previously selected camera."""
        with self._lock:
            self.latest = None
            self.error = None
        try:
            self.frames.get_nowait()
            self.frames.task_done()
        except queue.Empty:
            pass

    def _run(self) -> None:
        consecutive_errors = 0
        while not self._stop.is_set():
            try:
                frame = self.frames.get(timeout=0.1)
            except queue.Empty:
                continue

            height, width = frame.shape[:2]
            if width > self.inference_width:
                resized_height = round(height * self.inference_width / width)
                inference_frame = cv2.resize(
                    frame, (self.inference_width, resized_height)
                )
            else:
                inference_frame = frame

            started = time.perf_counter()
            try:
                response = self.client.infer(
                    inference_frame, model_id=self.model_id
                )
                result = InferenceResult(
                    predictions=response.get("predictions", []),
                    width=inference_frame.shape[1],
                    height=inference_frame.shape[0],
                    latency_ms=(time.perf_counter() - started) * 1000,
                )
                with self._lock:
                    self.latest = result
                    self.error = None
                consecutive_errors = 0
            except Exception as exc:  # SDK errors include HTTP and transport types.
                consecutive_errors += 1
                with self._lock:
                    retry_seconds = min(30, 2 ** min(consecutive_errors, 5))
                    self.error = f"{exc} (retrying in {retry_seconds}s)"
                self._stop.wait(retry_seconds)
            finally:
                self.frames.task_done()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect Rummikub tiles from a webcam in real time."
    )
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument(
        "--list-cameras",
        action="store_true",
        help="List available camera indices and exit",
    )
    parser.add_argument("--width", type=int, default=1280, help="Capture width")
    parser.add_argument("--height", type=int, default=720, help="Capture height")
    parser.add_argument(
        "--inference-width",
        type=int,
        default=640,
        help="Maximum width sent to Roboflow (default: 640)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.5,
        help="Seconds between inference submissions (default: 0.5)",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.4, help="Display threshold"
    )
    parser.add_argument(
        "--model-id",
        default=os.getenv("ROBOFLOW_MODEL_ID", DEFAULT_MODEL_ID),
        help=f"Roboflow model ID (default: {DEFAULT_MODEL_ID})",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.camera < 0:
        raise ValueError("--camera must be 0 or greater")
    if args.width <= 0 or args.height <= 0 or args.inference_width <= 0:
        raise ValueError("width and height values must be greater than 0")
    if args.interval <= 0:
        raise ValueError("--interval must be greater than 0")
    if not 0 <= args.confidence <= 1:
        raise ValueError("--confidence must be between 0 and 1")


def draw_predictions(
    frame: np.ndarray,
    result: InferenceResult | None,
    confidence_threshold: float,
) -> int:
    if result is None:
        return 0

    frame_height, frame_width = frame.shape[:2]
    scale_x = frame_width / result.width
    scale_y = frame_height / result.height
    count = 0

    for prediction in result.predictions:
        confidence = float(prediction.get("confidence", 0))
        if confidence < confidence_threshold:
            continue

        center_x = float(prediction["x"]) * scale_x
        center_y = float(prediction["y"]) * scale_y
        box_width = float(prediction["width"]) * scale_x
        box_height = float(prediction["height"]) * scale_y
        left = max(0, round(center_x - box_width / 2))
        top = max(0, round(center_y - box_height / 2))
        right = min(frame_width - 1, round(center_x + box_width / 2))
        bottom = min(frame_height - 1, round(center_y + box_height / 2))
        label = f'{prediction.get("class", "unknown")} {confidence:.0%}'

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 136), 2)
        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(text_height + 8, top)
        cv2.rectangle(
            frame,
            (left, label_top - text_height - 8),
            (left + text_width + 8, label_top),
            (0, 255, 136),
            -1,
        )
        cv2.putText(
            frame,
            label,
            (left + 4, label_top - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
        count += 1
    return count


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    capture = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not capture.isOpened():
        capture.release()
        capture = cv2.VideoCapture(index)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def find_available_cameras(max_index: int = 9) -> list[int]:
    available: list[int] = []
    previous_log_level = cv2.getLogLevel()
    cv2.setLogLevel(0)
    try:
        for index in range(max_index + 1):
            capture = open_camera(index, 640, 480)
            if capture.isOpened():
                ok, _ = capture.read()
                if ok:
                    available.append(index)
            capture.release()
    finally:
        cv2.setLogLevel(previous_log_level)
    return available


def switch_camera(
    current: cv2.VideoCapture,
    index: int,
    width: int,
    height: int,
) -> tuple[cv2.VideoCapture, bool]:
    replacement = open_camera(index, width, height)
    if not replacement.isOpened():
        replacement.release()
        return current, False
    ok, _ = replacement.read()
    if not ok:
        replacement.release()
        return current, False
    current.release()
    return replacement, True


def main() -> int:
    load_dotenv()
    args = parse_args()
    if args.list_cameras:
        available = find_available_cameras()
        if available:
            print("Available camera indices:", ", ".join(map(str, available)))
            return 0
        print("No available cameras found.")
        return 2
    try:
        validate_args(args)
        detector = AsyncDetector(args.model_id, args.inference_width)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    capture = open_camera(args.camera, args.width, args.height)
    if not capture.isOpened():
        print(f"Error: could not open camera {args.camera}")
        return 2

    detector.start()
    camera_index = args.camera
    known_cameras: list[int] | None = None
    camera_message = ""
    camera_message_until = 0.0
    last_submission = 0.0
    confidence = args.confidence
    fps = 0.0
    frame_count = 0
    fps_started = time.perf_counter()
    window_name = "Rummikub Live Detection"

    print(
        "Camera started. q/ESC: quit, c: next camera, 0-9: select camera, "
        "+/-: confidence."
    )
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                print("Error: failed to read a frame from the camera")
                return 3

            now = time.perf_counter()
            if now - last_submission >= args.interval and detector.submit(frame):
                last_submission = now

            result, error = detector.snapshot()
            detected = draw_predictions(frame, result, confidence)

            frame_count += 1
            elapsed = now - fps_started
            if elapsed >= 1:
                fps = frame_count / elapsed
                frame_count = 0
                fps_started = now

            latency = result.latency_ms if result else 0
            status = (
                f"CAM {camera_index} | FPS {fps:.1f} | tiles {detected} "
                f"| conf {confidence:.0%} "
                f"| API {latency:.0f} ms"
            )
            cv2.putText(
                frame,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            if error:
                message = f"API error: {error[:100]}"
                cv2.putText(
                    frame,
                    message,
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 0, 255),
                    2,
                    cv2.LINE_AA,
                )
            elif now < camera_message_until:
                cv2.putText(
                    frame,
                    camera_message,
                    (12, 56),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

            cv2.imshow(window_name, frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key in (ord("+"), ord("=")):
                confidence = min(1.0, confidence + 0.05)
            elif key in (ord("-"), ord("_")):
                confidence = max(0.0, confidence - 0.05)
            elif key == ord("c"):
                if known_cameras is None:
                    # The active camera can be locked by its current capture, so
                    # explicitly retain it while probing the other indices.
                    known_cameras = sorted(
                        set([camera_index, *find_available_cameras()])
                    )
                if len(known_cameras) < 2:
                    camera_message = "No other camera found"
                    camera_message_until = time.perf_counter() + 2
                    continue
                try:
                    position = known_cameras.index(camera_index)
                except ValueError:
                    position = -1
                target_index = known_cameras[(position + 1) % len(known_cameras)]
                capture, switched = switch_camera(
                    capture, target_index, args.width, args.height
                )
                if switched:
                    camera_index = target_index
                    detector.clear()
                    camera_message = f"Switched to camera {camera_index}"
                else:
                    known_cameras = None
                    camera_message = f"Could not open camera {target_index}"
                camera_message_until = time.perf_counter() + 2
            elif ord("0") <= key <= ord("9"):
                target_index = key - ord("0")
                if target_index == camera_index:
                    continue
                capture, switched = switch_camera(
                    capture, target_index, args.width, args.height
                )
                if switched:
                    camera_index = target_index
                    detector.clear()
                    known_cameras = None
                    camera_message = f"Switched to camera {camera_index}"
                else:
                    camera_message = f"Camera {target_index} is unavailable"
                camera_message_until = time.perf_counter() + 2
    finally:
        detector.stop()
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
