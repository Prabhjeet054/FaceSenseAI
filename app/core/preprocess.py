"""Face image preprocessing for attribute and recognition models."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np


GENDER_AGE_SIZE = (128, 128)
RECOGNITION_SIZE = (112, 112)
DEFAULT_CROP_MARGIN = 0.20


def crop_face(
    image: np.ndarray,
    bbox: Sequence[float],
    margin: float = DEFAULT_CROP_MARGIN,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop a face bounding box with a proportional margin.

    Returns the cropped image and the clipped ``(x, y, width, height)`` box
    used for the crop.
    """
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if image.ndim not in (2, 3):
        raise ValueError("image must be grayscale or a color image")
    if len(bbox) != 4:
        raise ValueError("bbox must contain x, y, width, and height")
    if margin < 0:
        raise ValueError("margin must be non-negative")

    x, y, width, height = (float(value) for value in bbox)
    if width <= 0 or height <= 0:
        raise ValueError("bbox width and height must be positive")

    image_height, image_width = image.shape[:2]
    margin_x = width * margin
    margin_y = height * margin
    left = max(0, int(np.floor(x - margin_x)))
    top = max(0, int(np.floor(y - margin_y)))
    right = min(image_width, int(np.ceil(x + width + margin_x)))
    bottom = min(image_height, int(np.ceil(y + height + margin_y)))

    if right <= left or bottom <= top:
        raise ValueError("bbox does not overlap the image")
    return image[top:bottom, left:right].copy(), (left, top, right - left, bottom - top)


def align_face(image: np.ndarray, eye_landmarks: Sequence[Sequence[float]]) -> np.ndarray:
    """Rotate a face so the line between its two eye landmarks is horizontal."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if len(eye_landmarks) != 2:
        raise ValueError("eye_landmarks must contain left and right eye points")

    left_eye = np.asarray(eye_landmarks[0], dtype=np.float32)
    right_eye = np.asarray(eye_landmarks[1], dtype=np.float32)
    if left_eye.shape != (2,) or right_eye.shape != (2,):
        raise ValueError("each eye landmark must contain x and y coordinates")

    delta_x = float(right_eye[0] - left_eye[0])
    delta_y = float(right_eye[1] - left_eye[1])
    if np.isclose(delta_x, 0.0) and np.isclose(delta_y, 0.0):
        raise ValueError("eye landmarks must not overlap")

    angle = float(np.degrees(np.arctan2(delta_y, delta_x)))
    center = tuple(np.mean([left_eye, right_eye], axis=0))
    rotation = cv2.getRotationMatrix2D(center, angle, 1.0)
    height, width = image.shape[:2]
    return cv2.warpAffine(
        image,
        rotation,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )


def resize_face(image: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Resize a face image to ``(width, height)``."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    if len(size) != 2 or any(dimension <= 0 for dimension in size):
        raise ValueError("size must contain two positive dimensions")
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def normalize_pixels(image: np.ndarray) -> np.ndarray:
    """Convert pixels to float32 values in the [0, 1] range."""
    if image is None or not isinstance(image, np.ndarray) or image.size == 0:
        raise ValueError("image must be a non-empty NumPy array")
    return image.astype(np.float32) / 255.0


def preprocess_face(
    image: np.ndarray,
    bbox: Sequence[float],
    landmarks: Sequence[Sequence[float]],
    model: str = "gender_age",
    margin: float = DEFAULT_CROP_MARGIN,
) -> np.ndarray:
    """Crop, align, resize, and normalize a detected face.

    ``landmarks`` must follow the detector convention: left eye, right eye,
    nose, left mouth corner, and right mouth corner.
    """
    if len(landmarks) < 2:
        raise ValueError("landmarks must contain both eye landmarks")

    cropped, (left, top, _, _) = crop_face(image, bbox, margin)
    adjusted_eyes = [
        (float(landmark[0]) - left, float(landmark[1]) - top)
        for landmark in landmarks[:2]
    ]
    aligned = align_face(cropped, adjusted_eyes)

    if model in {"gender_age", "gender", "age"}:
        size = GENDER_AGE_SIZE
    elif model in {"recognition", "face_recognition"}:
        size = RECOGNITION_SIZE
    else:
        raise ValueError("model must be 'gender_age' or 'recognition'")
    return normalize_pixels(resize_face(aligned, size))
