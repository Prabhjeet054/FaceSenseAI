from __future__ import annotations

import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.core.detector import FaceDetector
from app.core.embedder import FaceEmbedder
from app.core.model_runtime import TFLiteModel
from app.core.preprocess import (
    align_face,
    crop_face,
    normalize_pixels,
    preprocess_face,
    resize_face,
)
from app.models.age_model import predict_age_group
from app.models.emotion import EmotionClassifier


def test_detector_validates_configuration_and_detects_faces(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeDetector:
        def setInputSize(self, size: tuple[int, int]) -> None:
            assert size == (20, 20)

        def detect(self, image: np.ndarray) -> tuple[None, np.ndarray]:
            face = np.zeros(15, dtype=np.float32)
            face[:4] = [1, 2, 10, 12]
            face[4:14] = np.arange(10)
            face[14] = 0.9
            return None, np.asarray([face])

    monkeypatch.setattr(
        cv2.FaceDetectorYN,
        "create",
        lambda *args: FakeDetector(),
    )
    monkeypatch.setattr(cv2.dnn, "NMSBoxes", lambda *args: np.asarray([[0]]))
    model_path = tmp_path / "detector.onnx"
    model_path.touch()
    detector = FaceDetector(model_path=model_path)

    result = detector.detect(np.zeros((20, 20), dtype=np.uint8))

    assert result[0]["bbox"] == [1.0, 2.0, 10.0, 12.0]
    with pytest.raises(ValueError):
        FaceDetector(model_path=model_path, confidence_threshold=2)


def test_detector_color_conversion_and_empty_inputs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    class FakeDetector:
        def setInputSize(self, size: tuple[int, int]) -> None:
            pass

        def detect(self, image: np.ndarray) -> tuple[None, None]:
            return None, None

    monkeypatch.setattr(cv2.FaceDetectorYN, "create", lambda *args: FakeDetector())
    model_path = tmp_path / "detector.onnx"
    model_path.touch()
    detector = FaceDetector(model_path=model_path)
    assert detector.detect(None) == []
    assert detector.detect(np.zeros((10, 10, 4), dtype=np.uint8)) == []
    with pytest.raises(ValueError):
        detector.detect(np.zeros((10, 10, 2), dtype=np.uint8))


def test_embedder_validates_inputs_and_model_outputs() -> None:
    embedder = FaceEmbedder(model=types.SimpleNamespace(get_feat=lambda _: np.ones((1, 512))))
    with pytest.raises(ValueError):
        embedder.embed(np.zeros((10, 10), dtype=np.uint8))
    with pytest.raises(ValueError):
        embedder.embed([])

    bad_model = types.SimpleNamespace(get_feat=lambda _: np.ones((1, 3)))
    with pytest.raises(ValueError, match="unexpected shape"):
        FaceEmbedder(model=bad_model).embed(np.ones((10, 10, 3), dtype=np.uint8))

    failing_model = types.SimpleNamespace(
        get_feat=lambda _: (_ for _ in ()).throw(RuntimeError("failed"))
    )
    with pytest.raises(RuntimeError, match="failed to create"):
        FaceEmbedder(model=failing_model).embed(np.ones((10, 10, 3), dtype=np.uint8))


def test_embedder_loads_model_and_rejects_bad_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeModel:
        def prepare(self, ctx_id: int) -> None:
            assert ctx_id == 0

        def get_feat(self, faces: list[np.ndarray]) -> np.ndarray:
            return np.ones(512, dtype=np.float32)

    fake_zoo = types.SimpleNamespace(get_model=lambda name, **kwargs: FakeModel())
    monkeypatch.setitem(
        sys.modules,
        "insightface",
        types.SimpleNamespace(model_zoo=fake_zoo),
    )
    monkeypatch.setitem(sys.modules, "insightface.model_zoo", fake_zoo)
    loaded = FaceEmbedder(model_name="test", providers=["CPUExecutionProvider"])
    assert loaded.embed(np.ones((10, 10, 3), dtype=np.uint8)).shape == (512,)

    zero = FaceEmbedder(model=types.SimpleNamespace(get_feat=lambda _: np.zeros((1, 512))))
    with pytest.raises(ValueError, match="zero"):
        zero.embed(np.ones((10, 10, 3), dtype=np.uint8))


def test_embedder_accepts_batches_and_four_channel_faces() -> None:
    model = types.SimpleNamespace(
        get_feat=lambda faces: np.ones((len(faces), 512), dtype=np.float32)
    )
    embedder = FaceEmbedder(model=model)
    faces = np.ones((2, 10, 10, 4), dtype=np.uint8)
    result = embedder.embed(faces)
    assert result.shape == (2, 512)


@pytest.mark.parametrize(
    ("operation", "args"),
    [
        (crop_face, (np.ones((10, 10, 3)), [1, 2, 0, 3])),
        (resize_face, (np.ones((10, 10, 3)), (0, 2))),
        (normalize_pixels, (np.array([]),)),
        (align_face, (np.ones((10, 10, 3)), [[1, 1], [1, 1]])),
    ],
)
def test_preprocess_rejects_invalid_inputs(operation, args) -> None:
    with pytest.raises(ValueError):
        operation(*args)


def test_preprocess_supports_recognition_and_rejects_unknown_model() -> None:
    image = np.full((100, 100, 3), 255, dtype=np.uint8)
    landmarks = [[30, 30], [70, 30], [50, 50], [35, 70], [65, 70]]
    result = preprocess_face(image, [10, 10, 80, 80], landmarks, model="face_recognition")
    assert result.shape == (112, 112, 3)
    with pytest.raises(ValueError):
        preprocess_face(image, [10, 10, 80, 80], landmarks, model="unknown")


def test_age_group_boundaries_and_invalid_values() -> None:
    assert predict_age_group(12) == "child"
    assert predict_age_group(13) == "teen"
    assert predict_age_group(60) == "senior"
    with pytest.raises(ValueError):
        predict_age_group(-1)
    with pytest.raises(ValueError):
        predict_age_group(float("nan"))


def test_emotion_classifier_lazy_model_and_empty_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFer:
        def __init__(self, mtcnn: bool) -> None:
            assert mtcnn is False

        def detect_emotions(self, face: np.ndarray) -> list[dict[str, object]]:
            return [{"emotions": {"happy": 0.8}}]

    monkeypatch.setitem(sys.modules, "fer", types.SimpleNamespace(FER=FakeFer))
    classifier = EmotionClassifier()
    result = classifier.predict(np.ones((10, 10, 3), dtype=np.uint8))
    assert result == {"emotion": "happy", "emotion_confidence": 0.8}

    empty_model = EmotionClassifier(model=types.SimpleNamespace(detect_emotions=lambda _: []))
    with pytest.raises(ValueError, match="could not classify"):
        empty_model.predict(np.ones((10, 10, 3), dtype=np.uint8))


def test_tflite_model_predict_supports_dynamic_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeInterpreter:
        def __init__(self, model_path: str) -> None:
            self.shape = [1, 2]
            self.value = np.array([[3.0]], dtype=np.float32)

        def allocate_tensors(self) -> None:
            pass

        def get_input_details(self) -> list[dict[str, object]]:
            return [{"index": 1, "shape": np.asarray(self.shape)}]

        def get_output_details(self) -> list[dict[str, object]]:
            return [{"index": 2}]

        def resize_tensor_input(self, index: int, shape: tuple[int, ...]) -> None:
            self.shape = list(shape)

        def set_tensor(self, index: int, value: np.ndarray) -> None:
            assert value.shape == (2, 2)

        def invoke(self) -> None:
            pass

        def get_tensor(self, index: int) -> np.ndarray:
            return self.value

    fake_tf = types.SimpleNamespace(
        lite=types.SimpleNamespace(Interpreter=FakeInterpreter)
    )
    monkeypatch.setitem(sys.modules, "tensorflow", fake_tf)
    model = TFLiteModel("model.tflite")
    assert model.predict(np.ones((2, 2), dtype=np.float32)).shape == (1, 1)
