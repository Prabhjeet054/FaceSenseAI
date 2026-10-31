import numpy as np

from app.core.liveness import LivenessChecker


class FakeMesh:
    def __init__(self) -> None:
        self.calls = 0

    def process(self, image: np.ndarray) -> object:
        self.calls += 1
        return type("Result", (), {"multi_face_landmarks": None})()


class Landmark:
    def __init__(self, x: float, y: float) -> None:
        self.x = x
        self.y = y


class LandmarkMesh:
    def __init__(self, eye_aspects: list[float]) -> None:
        self.eye_aspects = eye_aspects
        self.calls = 0

    def process(self, image: np.ndarray) -> object:
        aspect = self.eye_aspects[self.calls]
        self.calls += 1
        landmarks = [Landmark(0.5, 0.5) for _ in range(478)]
        for indices in ((33, 160, 158, 133, 153, 144), (362, 385, 387, 263, 373, 380)):
            left, top_a, top_b, right, bottom_a, bottom_b = indices
            landmarks[left] = Landmark(0.0, 0.0)
            landmarks[right] = Landmark(1.0, 0.0)
            landmarks[top_a] = Landmark(0.25, aspect / 2)
            landmarks[top_b] = Landmark(0.75, aspect / 2)
            landmarks[bottom_a] = Landmark(0.75, -aspect / 2)
            landmarks[bottom_b] = Landmark(0.25, -aspect / 2)
        result = type("Result", (), {})()
        result.multi_face_landmarks = [type("Face", (), {"landmark": landmarks})()]
        return result


def textured_frames() -> list[np.ndarray]:
    frame = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
    return [np.repeat(frame[:, :, None], 3, axis=2)] * 3


def test_liveness_rejects_empty_sequence() -> None:
    checker = LivenessChecker(face_mesh=FakeMesh())

    try:
        checker.check([])
    except ValueError as error:
        assert "non-empty" in str(error)
    else:
        raise AssertionError("empty frame sequence should be rejected")


def test_liveness_returns_spoof_with_bounded_score_without_blink() -> None:
    checker = LivenessChecker(face_mesh=FakeMesh())
    frames = [np.zeros((32, 32, 3), dtype=np.uint8)] * 3

    result = checker.check(frames)

    assert result["status"] == "spoof"
    assert result["live"] is False
    assert 0 <= result["score"] <= 1


def test_blinking_landmark_sequence_is_live() -> None:
    checker = LivenessChecker(face_mesh=LandmarkMesh([0.6, 0.05, 0.6]))

    result = checker.check(textured_frames())

    assert result["status"] == "live"
    assert result["live"] is True
    assert result["blink_detected"] is True
    assert 0 <= result["score"] <= 1


def test_static_landmark_sequence_is_spoof() -> None:
    checker = LivenessChecker(face_mesh=LandmarkMesh([0.6, 0.6, 0.6]))

    result = checker.check(textured_frames())

    assert result["status"] == "spoof"
    assert result["live"] is False
    assert result["blink_detected"] is False
