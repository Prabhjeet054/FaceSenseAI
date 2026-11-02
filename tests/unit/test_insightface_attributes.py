from __future__ import annotations

import numpy as np

from app.core.insightface_attributes import InsightFaceAttributeModel


class FakeSession:
    def __init__(self) -> None:
        self.inputs = [type("Input", (), {"name": "data"})()]
        self.outputs = [type("Output", (), {"name": "fc1"})()]
        self.received: list[np.ndarray] = []

    def get_inputs(self) -> list[object]:
        return self.inputs

    def get_outputs(self) -> list[object]:
        return self.outputs

    def run(self, names: list[str], inputs: dict[str, np.ndarray]) -> list[np.ndarray]:
        del names
        image = inputs["data"]
        self.received.append(image)
        mean = float(image.mean()) / 255.0
        return [np.array([[0.2, -0.2, mean]], dtype=np.float32)]


def make_model(attribute: str) -> tuple[InsightFaceAttributeModel, FakeSession]:
    model = object.__new__(InsightFaceAttributeModel)
    session = FakeSession()
    model.attribute = attribute
    model.session = session
    model.input_name = "data"
    model.output_name = "fc1"
    return model, session


def test_age_model_preserves_per_face_values_and_original_pixel_scale() -> None:
    model, session = make_model("age")
    batch = np.stack(
        [
            np.zeros((32, 32, 3), dtype=np.float32),
            np.ones((32, 32, 3), dtype=np.float32),
        ]
    )

    predictions = model.predict(batch)

    assert predictions.shape == (2, 1)
    assert predictions[:, 0].tolist() == [0.0, 100.0]
    assert len(session.received) == 2
    assert float(session.received[1].max()) == 255.0


def test_gender_model_returns_probability_for_each_face() -> None:
    model, _ = make_model("gender")
    batch = np.zeros((2, 32, 32, 3), dtype=np.float32)

    predictions = model.predict(batch)

    assert predictions.shape == (2, 1)
    assert np.all((predictions >= 0) & (predictions <= 1))
