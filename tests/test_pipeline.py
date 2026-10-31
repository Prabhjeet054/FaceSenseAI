import numpy as np

from app.services.pipeline import FacePipeline


EXPECTED_KEYS = {
    "box",
    "name",
    "person_id",
    "match_score",
    "gender",
    "gender_confidence",
    "age",
    "age_group",
    "emotion",
    "emotion_confidence",
}


class GroupDetector:
    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
        if not image.any():
            return []
        return [
            {
                "bbox": [10, 10, 80, 80],
                "confidence": 0.95,
                "landmarks": [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]],
            },
            {
                "bbox": [110, 10, 80, 80],
                "confidence": 0.90,
                "landmarks": [[130, 30], [170, 30], [150, 50], [135, 70], [165, 70]],
            },
        ]


class FixedEmbedder:
    def embed(self, face: np.ndarray) -> np.ndarray:
        return np.ones(512, dtype=np.float32)


class FixedMatcher:
    def identify(self, embedding: np.ndarray) -> dict[str, object]:
        return {"name": "Known Person", "id": "person-1", "score": 0.85}


class FixedModel:
    def __init__(self, value: float) -> None:
        self.value = value

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        return np.full((len(batch), 1), self.value, dtype=np.float32)


class FixedEmotionModel:
    def predict(self, face: np.ndarray) -> dict[str, object]:
        assert face.ndim == 3
        return {"emotion": "happy", "emotion_confidence": 0.91}


def make_pipeline() -> FacePipeline:
    return FacePipeline(
        detector=GroupDetector(),
        embedder=FixedEmbedder(),
        matcher=FixedMatcher(),
        gender_model=FixedModel(0.75),
        age_model=FixedModel(25.0),
        emotion_model=FixedEmotionModel(),
    )


def test_group_photo_results_have_schema_and_valid_confidences() -> None:
    pipeline = make_pipeline()
    image = np.ones((120, 220, 3), dtype=np.uint8)

    results = pipeline.analyze(image)

    assert len(results) == 2
    for result in results:
        assert set(result) == EXPECTED_KEYS
        assert 0 <= result["match_score"] <= 1
        assert 0 <= result["gender_confidence"] <= 1
        assert result["emotion"] == "happy"
        assert 0 <= result["emotion_confidence"] <= 1


def test_image_with_no_faces_returns_empty_list() -> None:
    pipeline = make_pipeline()

    results = pipeline.analyze(np.zeros((120, 220, 3), dtype=np.uint8))

    assert results == []
