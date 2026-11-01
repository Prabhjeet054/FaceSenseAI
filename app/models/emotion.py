"""Seven-class facial emotion classification using the pretrained FER model."""

from __future__ import annotations

from typing import Any

import numpy as np


EMOTIONS = (
    "angry",
    "disgust",
    "fear",
    "happy",
    "neutral",
    "sad",
    "surprise",
)


class EmotionClassifier:
    """Classify a face crop with FER, loading the pretrained model on demand."""

    def __init__(self, model: Any | None = None) -> None:
        self._model = model

    def predict(self, face: np.ndarray) -> dict[str, Any]:
        """Return the most likely emotion and confidence for a BGR face crop."""
        if not isinstance(face, np.ndarray) or face.size == 0:
            raise ValueError("face must be a non-empty NumPy array")
        if face.ndim != 3 or face.shape[2] != 3:
            raise ValueError("face must be a color image with three channels")

        model = self._get_model()
        rgb_face = np.ascontiguousarray(face[:, :, ::-1])
        height, width = rgb_face.shape[:2]
        try:
            predictions = model.detect_emotions(
                rgb_face, face_rectangles=[(0, 0, width, height)]
            )
        except TypeError:
            predictions = model.detect_emotions(rgb_face)
        if not predictions:
            raise ValueError("FER could not classify the supplied face")
        emotions = predictions[0].get("emotions", {})
        scores = {name: float(emotions.get(name, 0.0)) for name in EMOTIONS}
        emotion = max(scores, key=scores.get)
        return {"emotion": emotion, "emotion_confidence": scores[emotion]}

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from fer import FER
            except ImportError as error:
                raise ImportError(
                    "Emotion detection requires the 'fer' package. "
                    "Install the pinned project dependencies first."
                ) from error
            self._model = FER(mtcnn=False)
        return self._model


__all__ = ["EMOTIONS", "EmotionClassifier"]
