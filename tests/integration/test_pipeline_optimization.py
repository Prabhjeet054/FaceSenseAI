from __future__ import annotations

import numpy as np

from app.services.pipeline import FacePipeline


class TwoFaceDetector:
    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
        return [
            {
                "bbox": [10, 10, 80, 80],
                "landmarks": [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]],
            },
            {
                "bbox": [110, 10, 80, 80],
                "landmarks": [[130, 30], [170, 30], [150, 50], [135, 70], [165, 70]],
            },
        ]


class CountingEmbedder:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def embed(self, faces: np.ndarray | list[np.ndarray]) -> np.ndarray:
        batch = [faces] if isinstance(faces, np.ndarray) and faces.ndim == 3 else list(faces)
        self.batch_sizes.append(len(batch))
        return np.ones((len(batch), 512), dtype=np.float32)


class CountingModel:
    def __init__(self, value: float) -> None:
        self.value = value
        self.batch_sizes: list[int] = []

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        self.batch_sizes.append(len(batch))
        return np.full((len(batch), 1), self.value, dtype=np.float32)


class FixedMatcher:
    def identify(self, embedding: np.ndarray) -> dict[str, object]:
        return {"name": "Known", "id": "person-1", "score": 0.9}


class FixedEmotion:
    def predict(self, face: np.ndarray) -> dict[str, object]:
        return {"emotion": "happy", "emotion_confidence": 0.9}


def make_pipeline(embedder: CountingEmbedder) -> tuple[FacePipeline, CountingModel, CountingModel]:
    gender = CountingModel(0.7)
    age = CountingModel(25.0)
    pipeline = FacePipeline(
        detector=TwoFaceDetector(),
        embedder=embedder,
        matcher=FixedMatcher(),
        gender_model=gender,
        age_model=age,
        emotion_model=FixedEmotion(),
        optimized=True,
    )
    return pipeline, gender, age


def test_optimized_pipeline_batches_models_and_reuses_embedding_cache() -> None:
    embedder = CountingEmbedder()
    pipeline, gender, age = make_pipeline(embedder)
    image = np.ones((120, 220, 3), dtype=np.uint8)

    assert len(pipeline.analyze(image)) == 2
    assert len(pipeline.analyze(image)) == 2

    assert embedder.batch_sizes == [2]
    assert gender.batch_sizes == [2, 2]
    assert age.batch_sizes == [2, 2]


def test_pipeline_handles_empty_images_and_single_only_embedders() -> None:
    embedder = CountingEmbedder()
    pipeline, _, _ = make_pipeline(embedder)
    assert pipeline.analyze(np.array([], dtype=np.uint8)) == []

    with np.testing.assert_raises(ValueError):
        FacePipeline(
            detector=TwoFaceDetector(),
            embedder=embedder,
            matcher=FixedMatcher(),
            gender_model=CountingModel(0.7),
            age_model=CountingModel(25.0),
            emotion_model=FixedEmotion(),
            embedding_cache_size=-1,
        )
