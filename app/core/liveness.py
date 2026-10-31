"""Liveness checks for short face-frame sequences."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


class LivenessChecker:
    """Combine MediaPipe eye-blink evidence with a texture anti-spoof score."""

    def __init__(
        self,
        face_mesh: Any | None = None,
        blink_threshold: float = 0.21,
        minimum_texture_score: float = 0.15,
    ) -> None:
        self.face_mesh = face_mesh
        self.blink_threshold = blink_threshold
        self.minimum_texture_score = minimum_texture_score

    def check(self, frames: list[np.ndarray]) -> dict[str, Any]:
        """Return ``live``/``spoof`` status and a score in ``[0, 1]``."""
        if not frames or any(not self._valid_frame(frame) for frame in frames):
            raise ValueError("liveness requires a non-empty sequence of color frames")

        texture_score = float(np.mean([self._texture_score(frame) for frame in frames]))
        blink_detected = self._blink_detected(frames)
        blink_score = 1.0 if blink_detected else 0.0
        score = 0.6 * blink_score + 0.4 * texture_score
        live = blink_detected and texture_score >= self.minimum_texture_score
        return {
            "status": "live" if live else "spoof",
            "live": live,
            "score": float(np.clip(score, 0.0, 1.0)),
            "blink_detected": blink_detected,
            "texture_score": texture_score,
        }

    def _blink_detected(self, frames: list[np.ndarray]) -> bool:
        mesh = self._get_face_mesh()
        ear_values: list[float] = []
        for frame in frames:
            result = mesh.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            landmarks = getattr(result, "multi_face_landmarks", None)
            if not landmarks:
                continue
            points = landmarks[0].landmark
            left = self._eye_aspect_ratio(points, (33, 160, 158, 133, 153, 144))
            right = self._eye_aspect_ratio(points, (362, 385, 387, 263, 373, 380))
            ear_values.append((left + right) / 2)
        if len(ear_values) < 2:
            return False
        return min(ear_values) < self.blink_threshold and max(ear_values) >= self.blink_threshold

    def _get_face_mesh(self) -> Any:
        if self.face_mesh is None:
            try:
                import mediapipe as mp
            except ImportError as error:
                raise ImportError(
                    "Liveness detection requires the 'mediapipe' package."
                ) from error
            self.face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
            )
        return self.face_mesh

    @staticmethod
    def _valid_frame(frame: np.ndarray) -> bool:
        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.size > 0
        )

    @staticmethod
    def _texture_score(frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        return float(np.clip(variance / 500.0, 0.0, 1.0))

    @staticmethod
    def _eye_aspect_ratio(points: Any, indices: tuple[int, ...]) -> float:
        p1, p2, p3, p4, p5, p6 = (
            np.array([points[index].x, points[index].y], dtype=np.float32)
            for index in indices
        )
        horizontal = np.linalg.norm(p1 - p4)
        if horizontal == 0:
            return 0.0
        return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * horizontal))


__all__ = ["LivenessChecker"]
