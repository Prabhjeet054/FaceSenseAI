"""End-to-end face analysis pipeline."""

from __future__ import annotations

from collections import OrderedDict
import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

import config
from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.insightface_attributes import InsightFaceAttributeModel
from app.core.preprocess import preprocess_face
from app.models.age_model import predict_age_group
from app.models.emotion import EmotionClassifier
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
        emotion_model: Any | None = None,
        gender_model_path: str | Path = config.GENDER_MODEL_PATH,
        age_model_path: str | Path = config.AGE_MODEL_PATH,
        optimized: bool = True,
        embedding_cache_size: int = 1024,
    ) -> None:
        if embedding_cache_size < 0:
            raise ValueError("embedding_cache_size must not be negative")
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()
        if matcher is None:
            registry = FaceRegistry(
                detector=self.detector,
                embedder=self.embedder,
            )
            matcher = FaceMatcher(registry)
        self.matcher = matcher
        self.gender_model = gender_model or self._load_attribute_model(
            gender_model_path, "gender"
        )
        self.age_model = age_model or self._load_attribute_model(age_model_path, "age")
        self.emotion_model = emotion_model or EmotionClassifier()
        self.optimized = optimized
        self._embedding_cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._embedding_cache_size = embedding_cache_size

    def analyze(self, image: np.ndarray) -> list[dict[str, Any]]:
        """Analyze all detected faces that meet the configured minimum size."""
        if image is None or not isinstance(image, np.ndarray) or image.size == 0:
            return []

        detections = [
            detection
            for detection in self.detector.detect(image)
            if self._meets_minimum_size(detection["bbox"])
        ]
        if self.optimized:
            return self._analyze_optimized(image, detections)
        return self._analyze_faces(image, detections)

    def _analyze_faces(
        self, image: np.ndarray, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for detection in detections:
            bbox = detection["bbox"]
            recognition_face = preprocess_face(
                image, bbox, detection["landmarks"], model="recognition"
            )
            embedding = (
                self._cached_embedding(recognition_face)
                if self.optimized
                else np.asarray(self.embedder.embed(recognition_face), dtype=np.float32)
            )
            match = self.matcher.identify(embedding)
            if isinstance(match, list):
                raise TypeError("matcher.identify must return a single match")

            attribute_face = preprocess_face(
                image, bbox, detection["landmarks"], model="gender_age"
            )
            batch = np.expand_dims(
                self._attribute_input(attribute_face, self.gender_model), axis=0
            )
            gender_probability = float(
                np.asarray(self.gender_model.predict(batch, verbose=0)).reshape(-1)[0]
            )
            age = float(
                np.asarray(
                    self.age_model.predict(
                        np.expand_dims(
                            self._attribute_input(attribute_face, self.age_model),
                            axis=0,
                        ),
                        verbose=0,
                    )
                ).reshape(-1)[0]
            )
            face_crop = self._crop_face(image, bbox)
            emotion = self.emotion_model.predict(face_crop)
            gender = "female" if gender_probability >= 0.5 else "male"
            results.append(
                {
                    "box": [float(value) for value in bbox],
                    "name": match["name"],
                    "person_id": match["id"],
                    "match_score": float(match["score"]),
                    "gender": gender,
                    "gender_confidence": max(
                        gender_probability, 1.0 - gender_probability
                    ),
                    "age": age,
                    "age_group": predict_age_group(age),
                    "emotion": emotion["emotion"],
                    "emotion_confidence": float(emotion["emotion_confidence"]),
                }
            )
        return results

    def _analyze_optimized(
        self, image: np.ndarray, detections: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        if not detections:
            return []

        recognition_faces = [
            preprocess_face(
                image, detection["bbox"], detection["landmarks"], model="recognition"
            )
            for detection in detections
        ]
        embeddings: list[np.ndarray | None] = [None] * len(recognition_faces)
        uncached_faces: list[np.ndarray] = []
        uncached_indices: list[int] = []
        for index, face in enumerate(recognition_faces):
            cached = self._embedding_cache.get(self._embedding_key(face))
            if cached is None:
                uncached_faces.append(face)
                uncached_indices.append(index)
            else:
                self._embedding_cache.move_to_end(self._embedding_key(face))
                embeddings[index] = cached
        if uncached_faces:
            computed = self._embed_batch(uncached_faces)
            if computed.ndim == 1:
                computed = computed.reshape(1, -1)
            if computed.shape[0] != len(uncached_faces):
                raise ValueError("embedder returned an unexpected batch size")
            for index, embedding in zip(uncached_indices, computed):
                self._cache_embedding(recognition_faces[index], embedding)
                embeddings[index] = embedding

        attribute_faces = [
            self._attribute_input(
                preprocess_face(
                    image,
                    detection["bbox"],
                    detection["landmarks"],
                    model="gender_age",
                ),
                self.gender_model,
            )
            for detection in detections
        ]
        attribute_batch = np.stack(attribute_faces, axis=0)
        gender_probabilities = np.asarray(
            self.gender_model.predict(attribute_batch, verbose=0)
        ).reshape(-1)
        age_batch = np.stack(
            [
                self._attribute_input(
                    preprocess_face(
                        image,
                        detection["bbox"],
                        detection["landmarks"],
                        model="gender_age",
                    ),
                    self.age_model,
                )
                for detection in detections
            ],
            axis=0,
        )
        ages = np.asarray(self.age_model.predict(age_batch, verbose=0)).reshape(-1)
        if len(gender_probabilities) != len(detections) or len(ages) != len(detections):
            raise ValueError("attribute models returned an unexpected batch size")

        results: list[dict[str, Any]] = []
        for index, detection in enumerate(detections):
            embedding = embeddings[index]
            if embedding is None:
                raise RuntimeError("embedding batch contained an empty result")
            match = self.matcher.identify(embedding)
            if isinstance(match, list):
                raise TypeError("matcher.identify must return a single match")
            gender_probability = float(gender_probabilities[index])
            age = float(ages[index])
            emotion = self.emotion_model.predict(
                self._crop_face(image, detection["bbox"])
            )
            results.append(
                self._result(
                    detection,
                    match,
                    gender_probability,
                    age,
                    emotion,
                )
            )
        return results

    def _embed_batch(self, faces: list[np.ndarray]) -> np.ndarray:
        computed = np.asarray(self.embedder.embed(faces), dtype=np.float32)
        if computed.ndim == 2 and computed.shape[0] == len(faces):
            return computed
        if len(faces) == 1 and computed.ndim == 1:
            return computed.reshape(1, -1)
        # Keep custom embedders compatible while FaceEmbedder uses one batch call.
        return np.stack(
            [
                np.asarray(self.embedder.embed(face), dtype=np.float32)
                for face in faces
            ]
        )

    def _cached_embedding(self, face: np.ndarray) -> np.ndarray:
        key = self._embedding_key(face)
        cached = self._embedding_cache.get(key)
        if cached is not None:
            self._embedding_cache.move_to_end(key)
            return cached
        embedding = np.asarray(self.embedder.embed(face), dtype=np.float32)
        self._cache_embedding(face, embedding)
        return embedding

    def _cache_embedding(self, face: np.ndarray, embedding: np.ndarray) -> None:
        if self._embedding_cache_size == 0:
            return
        key = self._embedding_key(face)
        self._embedding_cache[key] = embedding.copy()
        self._embedding_cache.move_to_end(key)
        while len(self._embedding_cache) > self._embedding_cache_size:
            self._embedding_cache.popitem(last=False)

    @staticmethod
    def _embedding_key(face: np.ndarray) -> str:
        contiguous = np.ascontiguousarray(face)
        digest = hashlib.blake2b(digest_size=16)
        digest.update(contiguous.tobytes())
        digest.update(str(contiguous.shape).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _result(
        detection: dict[str, Any],
        match: dict[str, Any],
        gender_probability: float,
        age: float,
        emotion: dict[str, Any],
    ) -> dict[str, Any]:
        return {
            "box": [float(value) for value in detection["bbox"]],
            "name": match["name"],
            "person_id": match["id"],
            "match_score": float(match["score"]),
            "gender": "female" if gender_probability >= 0.5 else "male",
            "gender_confidence": max(gender_probability, 1.0 - gender_probability),
            "age": age,
            "age_group": predict_age_group(age),
            "emotion": emotion["emotion"],
            "emotion_confidence": float(emotion["emotion_confidence"]),
        }

    @staticmethod
    def _attribute_input(face: np.ndarray, model: Any) -> np.ndarray:
        """Match channel order to the attribute model's training contract."""
        if isinstance(model, InsightFaceAttributeModel):
            return face
        return face[..., ::-1].copy()

    @staticmethod
    def _load_model(model_path: str | Path) -> Any:
        path = Path(model_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Model file not found: {path}. Train the required model first."
            )
        return tf.keras.models.load_model(path, compile=False)

    @staticmethod
    def _load_attribute_model(model_path: str | Path, attribute: str) -> Any:
        path = Path(model_path)
        if path.is_file():
            return tf.keras.models.load_model(path, compile=False)
        bundled_path = Path.home() / ".insightface" / "models" / "genderage.onnx"
        if bundled_path.is_file():
            return InsightFaceAttributeModel(bundled_path, attribute)
        raise FileNotFoundError(
            f"Model file not found: {path}. Train the required model first."
        )

    @staticmethod
    def _meets_minimum_size(bbox: Any) -> bool:
        if len(bbox) != 4:
            raise ValueError("detector bounding boxes must contain four values")
        _, _, width, height = (float(value) for value in bbox)
        return width >= config.FACE_MIN_SIZE and height >= config.FACE_MIN_SIZE

    @staticmethod
    def _crop_face(image: np.ndarray, bbox: Any) -> np.ndarray:
        x, y, width, height = (int(round(float(value))) for value in bbox)
        image_height, image_width = image.shape[:2]
        left = max(0, x)
        top = max(0, y)
        right = min(image_width, x + width)
        bottom = min(image_height, y + height)
        if right <= left or bottom <= top:
            raise ValueError("face bounding box does not overlap the image")
        return image[top:bottom, left:right]


__all__ = ["FacePipeline"]
