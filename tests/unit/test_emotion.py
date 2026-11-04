import numpy as np
import pytest

from app.models.emotion import EMOTIONS, EmotionClassifier


class FakeFER:
    def detect_emotions(self, image: np.ndarray) -> list[dict[str, dict[str, float]]]:
        assert image.shape == (32, 32, 3)
        assert np.array_equal(image[:, :, 0], image[:, :, 1])
        assert np.array_equal(image[:, :, 1], image[:, :, 2])
        return [
            {
                "emotions": {
                    "angry": 0.02,
                    "disgust": 0.01,
                    "fear": 0.03,
                    "happy": 0.88,
                    "neutral": 0.02,
                    "sad": 0.02,
                    "surprise": 0.02,
                }
            }
        ]


def test_emotion_classifier_returns_one_of_seven_emotions() -> None:
    result = EmotionClassifier(model=FakeFER()).predict(
        np.zeros((32, 32, 3), dtype=np.uint8)
    )

    assert tuple(EMOTIONS) == (
        "angry",
        "disgust",
        "fear",
        "happy",
        "neutral",
        "sad",
        "surprise",
    )
    assert result["emotion"] in EMOTIONS
    assert 0 <= result["emotion_confidence"] <= 1
    assert result == {"emotion": "happy", "emotion_confidence": pytest.approx(0.88)}


def test_emotion_classifier_rejects_empty_faces() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        EmotionClassifier(model=FakeFER()).predict(np.empty((0, 0, 3), dtype=np.uint8))
