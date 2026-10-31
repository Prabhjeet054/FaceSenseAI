"""Drawing helpers for face-analysis overlays."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import cv2
import numpy as np


MALE_COLOR = (255, 0, 0)
FEMALE_COLOR = (180, 105, 255)
UNKNOWN_COLOR = (128, 128, 128)
TEXT_COLOR = (255, 255, 255)


def draw_face(
    image: np.ndarray,
    result: Mapping[str, object],
    corner_radius: int = 12,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a rounded face box and readable identity/attribute label."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("image must be a BGR color image")
    if corner_radius < 0 or thickness <= 0:
        raise ValueError("corner_radius must be non-negative and thickness positive")

    x, y, width, height = _box(result.get("box"))
    color = _color_for_result(result)
    output = image.copy()
    _draw_rounded_rectangle(
        output,
        (x, y),
        (x + width, y + height),
        color,
        corner_radius,
        thickness,
    )

    label = _format_label(result)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    text_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        label, font, font_scale, text_thickness
    )
    image_height, image_width = output.shape[:2]
    label_x = max(0, min(x, image_width - text_width - 12))
    label_y = y - 8
    if label_y - text_height - baseline < 0:
        label_y = min(image_height - baseline - 2, y + height + text_height + baseline + 8)
    label_y = max(text_height + baseline + 2, label_y)

    background_top = max(0, label_y - text_height - baseline - 6)
    background_bottom = min(image_height, label_y + 4)
    background_left = max(0, label_x - 6)
    background_right = min(image_width, label_x + text_width + 6)
    cv2.rectangle(
        output,
        (background_left, background_top),
        (background_right, background_bottom),
        color,
        thickness=cv2.FILLED,
    )
    cv2.putText(
        output,
        label,
        (label_x, label_y),
        font,
        font_scale,
        TEXT_COLOR,
        text_thickness,
        cv2.LINE_AA,
    )
    return output


def draw_faces(image: np.ndarray, results: Sequence[Mapping[str, object]]) -> np.ndarray:
    """Draw overlays for every face result."""
    output = image.copy()
    for result in results:
        output = draw_face(output, result)
    return output


def _format_label(result: Mapping[str, object]) -> str:
    name = str(result.get("name", "Unknown"))
    gender = str(result.get("gender", "unknown")).capitalize()
    confidence = float(result.get("gender_confidence", 0.0)) * 100
    age = float(result.get("age", 0.0))
    return f"{name} | {gender} ({confidence:.0f}%) | Age ~{age:.0f}"


def _color_for_result(result: Mapping[str, object]) -> tuple[int, int, int]:
    if str(result.get("name", "")).lower() == "unknown":
        return UNKNOWN_COLOR
    gender = str(result.get("gender", "")).lower()
    if gender == "male":
        return MALE_COLOR
    if gender == "female":
        return FEMALE_COLOR
    return UNKNOWN_COLOR


def _box(value: object) -> tuple[int, int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 4:
        raise ValueError("result['box'] must contain x, y, width, and height")
    x, y, width, height = (int(round(float(item))) for item in value)
    if width <= 0 or height <= 0:
        raise ValueError("result['box'] width and height must be positive")
    return x, y, width, height


def _draw_rounded_rectangle(
    image: np.ndarray,
    top_left: tuple[int, int],
    bottom_right: tuple[int, int],
    color: tuple[int, int, int],
    radius: int,
    thickness: int,
) -> None:
    x1, y1 = top_left
    x2, y2 = bottom_right
    radius = min(radius, max(0, (x2 - x1) // 2), max(0, (y2 - y1) // 2))
    if radius == 0:
        cv2.rectangle(image, top_left, bottom_right, color, thickness)
        return

    cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
    cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
    cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
    cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)
    for center, start, end in (
        ((x1 + radius, y1 + radius), 180, 270),
        ((x2 - radius, y1 + radius), 270, 360),
        ((x2 - radius, y2 - radius), 0, 90),
        ((x1 + radius, y2 - radius), 90, 180),
    ):
        cv2.ellipse(image, center, (radius, radius), 0, start, end, color, thickness)
