"""High-visibility, color-aware detection overlays."""

from __future__ import annotations

import cv2
import numpy as np


COLOR_RGB = {
    "black": (20, 20, 20),
    "red": (235, 45, 55),
    "orange": (255, 145, 0),
    "blue": (25, 105, 235),
    "joker": (180, 45, 200),
}


def class_color(class_name: str, channel_order: str = "RGB") -> tuple[int, int, int]:
    """Return a stable tile color in RGB or BGR channel order."""
    compact = "".join(character for character in class_name.lower() if character.isalnum())
    family = next((color for color in ("black", "red", "orange", "blue") if compact.startswith(color)), "joker")
    rgb = COLOR_RGB.get(family, COLOR_RGB["joker"])
    return rgb if channel_order.upper() == "RGB" else rgb[::-1]


def draw_box_label(
    canvas: np.ndarray,
    xyxy: tuple[int, int, int, int],
    text: str,
    class_name: str,
    channel_order: str = "RGB",
) -> None:
    """Draw a thick outlined box and a large, contrast-safe label."""
    image_height, image_width = canvas.shape[:2]
    x1, y1, x2, y2 = xyxy
    x1, x2 = sorted((max(0, min(image_width - 1, x1)), max(0, min(image_width - 1, x2))))
    y1, y2 = sorted((max(0, min(image_height - 1, y1)), max(0, min(image_height - 1, y2))))
    color = class_color(class_name, channel_order)
    minimum_side = min(image_height, image_width)
    line_width = max(4, round(minimum_side / 220))
    font_scale = max(0.82, min(1.35, minimum_side / 900))
    text_thickness = max(2, round(line_width / 2))

    # A contrasting halo keeps black visible on dark tiles and colors visible on clutter.
    compact_name = "".join(character for character in class_name.lower() if character.isalnum())
    halo = (245, 245, 245) if compact_name.startswith("black") else (25, 25, 25)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), halo, line_width + 4, cv2.LINE_AA)
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, line_width, cv2.LINE_AA)

    (text_width, text_height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
    )
    pad_x, pad_y = max(7, line_width + 2), max(6, line_width + 1)
    label_width = min(image_width, text_width + pad_x * 2)
    label_left = x1 if x1 + label_width <= image_width else max(0, x2 - label_width)
    label_height = text_height + baseline + pad_y * 2
    label_top = y1 - label_height
    if label_top < 0:
        label_top = y1
    label_bottom = min(image_height - 1, label_top + label_height)
    label_right = min(image_width - 1, label_left + label_width)
    cv2.rectangle(canvas, (label_left, label_top), (label_right, label_bottom), halo, -1)
    inset = max(2, round(line_width / 2))
    cv2.rectangle(
        canvas,
        (min(label_right, label_left + inset), min(label_bottom, label_top + inset)),
        (max(label_left, label_right - inset), max(label_top, label_bottom - inset)),
        color,
        -1,
    )
    text_color = (10, 10, 10) if compact_name.startswith("orange") else (255, 255, 255)
    text_x = min(label_right - 1, label_left + pad_x)
    text_y = min(label_bottom - pad_y, label_top + pad_y + text_height)
    cv2.putText(
        canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX,
        font_scale, text_color, text_thickness, cv2.LINE_AA,
    )


def draw_invalid_highlights(
    canvas: np.ndarray,
    boxes: list[dict],
    violation_sets: dict[int, int],
    channel_order: str = "RGB",
) -> np.ndarray:
    """Emphasize rule-invalid detections with tint, red border, cross, and badge."""
    output = canvas.copy()
    image_height, image_width = output.shape[:2]
    red = (255, 30, 35) if channel_order.upper() == "RGB" else (35, 30, 255)
    white = (255, 255, 255)
    for box_index, set_index in violation_sets.items():
        if not 0 <= box_index < len(boxes):
            continue
        box = boxes[box_index]
        x1, x2 = sorted((int(box["x1"]), int(box["x2"])))
        y1, y2 = sorted((int(box["y1"]), int(box["y2"])))
        x1, x2 = max(0, x1), min(image_width - 1, x2)
        y1, y2 = max(0, y1), min(image_height - 1, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        overlay = output.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), red, -1)
        output = cv2.addWeighted(overlay, 0.22, output, 0.78, 0)
        line_width = max(5, round(min(image_height, image_width) / 160))
        cv2.rectangle(output, (x1, y1), (x2, y2), white, line_width + 5, cv2.LINE_AA)
        cv2.rectangle(output, (x1, y1), (x2, y2), red, line_width, cv2.LINE_AA)
        cv2.line(output, (x1, y1), (x2, y2), white, line_width + 4, cv2.LINE_AA)
        cv2.line(output, (x2, y1), (x1, y2), white, line_width + 4, cv2.LINE_AA)
        cv2.line(output, (x1, y1), (x2, y2), red, line_width, cv2.LINE_AA)
        cv2.line(output, (x2, y1), (x1, y2), red, line_width, cv2.LINE_AA)

        badge = f"ERROR SET {set_index}"
        font_scale = max(0.65, min(1.05, min(image_height, image_width) / 1000))
        thickness = max(2, line_width // 2)
        (text_width, text_height), baseline = cv2.getTextSize(
            badge, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        badge_width = min(image_width, text_width + 14)
        badge_left = x1 if x1 + badge_width <= image_width else max(0, x2 - badge_width)
        badge_bottom = min(image_height - 1, y2)
        badge_top = max(0, badge_bottom - text_height - baseline - 12)
        cv2.rectangle(output, (badge_left, badge_top), (badge_left + badge_width, badge_bottom), white, -1)
        cv2.rectangle(
            output,
            (badge_left + 2, badge_top + 2),
            (min(image_width - 1, badge_left + badge_width - 2), badge_bottom - 2),
            red,
            -1,
        )
        cv2.putText(
            output, badge, (badge_left + 7, badge_bottom - baseline - 5),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, white, thickness, cv2.LINE_AA,
        )
    return output
