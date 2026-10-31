"""End-to-end face analysis pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import config
from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.preprocess import preprocess_face
from app.models.age_model import predict_age_group
from app.services.matcher import FaceMatcher
from app.services.registry import FaceRegistry


class FacePipeline:
    """Detect, identify, and estimate attributes for every sufficiently large face."""

    def __init__(
        self,
        detector: Any | None = None,
        embedder: Any | None = None,
        matcher: FaceMatcher | Any | None = None,
        gender_model: Any | None = None,
        age_model: Any | None = None,
        gender_model_path: str | Path = config.GENDER_MODEL_PATH,
        age_model_path: str | Path = config.AGE_MODEL_PATH,
    ) -> None:
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        if matcher is None:
            registry = FaceRegistry(
                detector=self.detector,
                embedder=self.embedder,
            )
            matcher = FaceMatcher(registry)
        self.matcher = matcher
        self.gender_model = gender_model or self._load_model(gender_model_path)
        self.age_model = age_model or self._load_model(age_model_path)

    def analyze(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Analyze all detected faces that meet the configured minimum size."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        results: list[dict[str, Any]] = []
        for detection in self.detector.detect(image):
            bbox = detection["bbox"]
            if not self._meets_minimum_size(bbox):
                continue

            recognition_face = preprocess_face(
                image, bbox, detection["landmarks"], model="recognition"
            )
            embedding = self.embedder.embed(recognition_face)
            match = self.matcher.identify(embedding)
            if isinstance(match, list):
                raise TypeError("matcher.identify must return a single match")

            attribute_face = preprocess_face(
                image, bbox, detection["landmarks"], model="gender_age"
            )
            batch = np.expand_dims(attribute_face, axis=0)
            gender_probability = float(
                np.asarray(self.gender_model.predict(batch, verbose=0)).reshape(-1)[0]
            )
            age = float(
                np.asarray(self.age_model.predict(batch, verbose=0)).reshape(-1)[0]
            )
            gender = "female" if gender_probability >= 0.5 else "male"
            results.append(
                {
                    "box": [float(value) for value in bbox],
                    "name": match["name"],
                    "match_score": float(match["score"]),
                    "gender": gender,
                    "gender_confidence": max(
                        gender_probability, 1.0 - gender_probability
                    ),
                    "age": age,
                    "age_group": predict_age_group(age),
                }
            )
        return results

    @staticmethod
    def _load_model(model_path: str | Path) -> Any:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}. Train the required model first."
            )
        return tf.keras.models.load_model(path, compile=False)

    @staticmethod
    def _meets_minimum_size(bbox: Any) -> bool:
        if len(bbox) != 4:
            raise ValueError("detector bounding boxes must contain four values")
        _, _, width, height = (float(value) for value in bbox)
        return width >= config.FACE_MIN_SIZE and height >= config.FACE_MIN_SIZE


__all__ = ["FacePipeline"]
