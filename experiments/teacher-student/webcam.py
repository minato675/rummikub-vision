"""Real-time webcam inference against the local Roboflow Inference server."""

from __future__ import annotations

import argparse
import time

import cv2

from rummikub_inference import build_client, draw_predictions, infer


def open_camera(index: int, width: int, height: int) -> cv2.VideoCapture:
    camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open camera {index}")
    return camera


def available_cameras(max_index: int = 8) -> list[int]:
    found: list[int] = []
    for index in range(max_index + 1):
        camera = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if camera.isOpened():
            ok, _ = camera.read()
            if ok:
                found.append(index)
        camera.release()
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="Local Roboflow Rummikub webcam detection")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--list-cameras", action="store_true")
    parser.add_argument("--max-camera", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--confidence", type=float, default=0.35)
    args = parser.parse_args()

    cameras = available_cameras(args.max_camera)
    if args.list_cameras:
        print("Available cameras:", cameras if cameras else "none")
        return 0
    if not cameras:
        raise RuntimeError("No webcam was found")
    current = args.camera if args.camera in cameras else cameras[0]
    client, settings = build_client(args.confidence)
    camera = open_camera(current, args.width, args.height)
    print(f"Local model {settings.model_id} via {settings.server_url}")
    print("Keys: q/Esc quit, [ previous camera, ] next camera")

    try:
        while True:
            ok, frame = camera.read()
            if not ok:
                continue
            started = time.perf_counter()
            result = infer(client, settings.model_id, frame)
            latency_ms = (time.perf_counter() - started) * 1000
            canvas = draw_predictions(frame, result)
            status = f"camera={current} objects={len(result.get('predictions', []))} latency={latency_ms:.1f}ms"
            cv2.putText(canvas, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (20, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Rummikub - Local Roboflow Inference", canvas)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key in (ord("["), ord("]")) and len(cameras) > 1:
                position = cameras.index(current)
                offset = -1 if key == ord("[") else 1
                target = cameras[(position + offset) % len(cameras)]
                replacement = open_camera(target, args.width, args.height)
                camera.release()
                camera, current = replacement, target
    finally:
        camera.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

