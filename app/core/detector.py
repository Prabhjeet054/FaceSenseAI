"""Face detection using OpenCV's YuNet DNN detector."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

import config


class FaceDetector:
    """Detect faces and five facial landmarks in an image."""

    def __init__(
        self,
        model_path: str | Path = config.FACE_DETECTION_MODEL_PATH,
        confidence_threshold: float = config.FACE_DETECTION_CONFIDENCE,
        nms_threshold: float = config.FACE_NMS_THRESHOLD,
        input_size: tuple[int, int] = (320, 320),
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                f"Face detection model not found: {self.model_path}. "
                "Download the YuNet ONNX model and place it in the models directory."
            )
        if not 0 <= confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be between 0 and 1")
        if not 0 <= nms_threshold <= 1:
            raise ValueError("nms_threshold must be between 0 and 1")

        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.detector = cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            input_size,
            confidence_threshold,
            nms_threshold,
        )

    def detect(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Return bounding boxes, confidence scores, and five landmarks."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        image = self._as_color(image)
        height, width = image.shape[:2]
        self.detector.setInputSize((width, height))
        _, faces = self.detector.detect(image)
        if faces is None or len(faces) == 0:
            return []

        candidates = np.asarray(faces, dtype=np.float32)
        boxes = candidates[:, :4].tolist()
        scores = candidates[:, 14].tolist()
        kept_indices = cv2.dnn.NMSBoxes(
            boxes,
            scores,
            self.confidence_threshold,
            self.nms_threshold,
        )
        indices = np.asarray(kept_indices).reshape(-1).tolist()

        detections: list[dict[str, Any]] = []
        for index in indices:
            face = candidates[int(index)]
            detections.append(
                {
                    "bbox": [float(value) for value in face[:4]],
                    "confidence": float(face[14]),
                    "landmarks": [
                        [float(face[offset]), float(face[offset + 1])]
                        for offset in range(4, 14, 2)
                    ],
                }
            )
        return detections

    @staticmethod
    def _as_color(image: np.ndarray) -> np.ndarray:
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.ndim != 3:
            raise ValueError("image must be a grayscale or color image")
        if image.shape[2] == 1:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        if image.shape[2] == 3:
            return image
        raise ValueError("image must have 1, 3, or 4 channels")
