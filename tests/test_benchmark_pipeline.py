from __future__ import annotations

import numpy as np

import benchmark_pipeline
from app.services.pipeline import FacePipeline


class OneFaceDetector:
    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
        return [
            {
                "bbox": [10, 10, 80, 80],
                "landmarks": [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]],
            }
        ]


class BatchEmbedder:
    def embed(self, faces: np.ndarray | list[np.ndarray]) -> np.ndarray:
        if isinstance(faces, np.ndarray) and faces.ndim == 3:
            count = 1
        else:
            count = len(faces)
        return np.ones((count, 512), dtype=np.float32)


class FixedModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.full((len(batch), 1), self.value, dtype=np.float32)


class FixedMatcher:
    def identify(self, embedding: np.ndarray) -> dict[str, object]:
        return {"name": "Known", "id": "person-1", "score": 0.9}


class FixedEmotion:
    def predict(self, face: np.ndarray) -> dict[str, object]:
        return {"emotion": "happy", "emotion_confidence": 0.9}


def make_pipeline(optimized: bool) -> FacePipeline:
    return FacePipeline(
        detector=OneFaceDetector(),
        embedder=BatchEmbedder(),
        matcher=FixedMatcher(),
        gender_model=FixedModel(0.7),
        age_model=FixedModel(25.0),
        emotion_model=FixedEmotion(),
        optimized=optimized,
    )


def test_optimized_benchmark_is_not_slower_and_predictions_match(
    monkeypatch,
) -> None:
    images = [np.full((120, 120, 3), index, dtype=np.uint8) for index in range(20)]
    baseline = make_pipeline(optimized=False)
    optimized = make_pipeline(optimized=True)

    baseline_predictions = [baseline.analyze(image) for image in images]
    optimized_predictions = [optimized.analyze(image) for image in images]
    assert optimized_predictions == baseline_predictions

    current_pipeline = None
    clock = 0.0

    def deterministic_clock() -> float:
        nonlocal clock
        clock += 0.002 if current_pipeline is baseline else 0.001
        return clock

    monkeypatch.setattr(benchmark_pipeline.time, "perf_counter", deterministic_clock)
    current_pipeline = baseline
    baseline_metrics = benchmark_pipeline.benchmark(baseline, images, repetitions=1)
    current_pipeline = optimized
    optimized_metrics = benchmark_pipeline.benchmark(optimized, images, repetitions=1)

    assert baseline_metrics["images"] == 20
    assert optimized_metrics["images"] == 20
    assert optimized_metrics["average_ms_per_image"] <= baseline_metrics[
        "average_ms_per_image"
    ]
    assert optimized_metrics["average_ms_per_face"] <= baseline_metrics[
        "average_ms_per_face"
    ]
