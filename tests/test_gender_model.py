import pytest

tf = pytest.importorskip("tensorflow")

from app.models.gender_model import build_model


def test_gender_model_shapes_and_prediction_range() -> None:
    model = build_model()

    assert model.input_shape[1:] == (128, 128, 3)
    assert model.output_shape == (None, 1)

    random_faces = tf.random.uniform(
        shape=(1, 128, 128, 3),
        minval=0,
        maxval=1,
        seed=42,
    )
    predictions = model(random_faces, training=False).numpy()

    assert predictions.shape == (1, 1)
    assert ((predictions >= 0) & (predictions <= 1)).all()
