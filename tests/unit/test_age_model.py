import pytest

tf = pytest.importorskip("tensorflow")

from app.models.age_model import build_model, calibrate_age, predict_age_group


def test_calibrate_age_corrects_older_face_bias() -> None:
    assert calibrate_age(60) == pytest.approx(59.79)
    assert calibrate_age(70) == pytest.approx(67.37)


def test_calibrate_age_rejects_invalid_values() -> None:
    with pytest.raises(ValueError):
        calibrate_age(-1)


def test_age_model_returns_one_non_negative_prediction_per_image() -> None:
    model = build_model(weights=None)
    images = tf.random.uniform((3, 128, 128, 3), seed=42)

    predictions = model(images, training=False).numpy()

    assert predictions.shape == (3, 1)
    assert (predictions >= 0).all()


@pytest.mark.parametrize(
    ("age", "expected_group"),
    [
        (5, "child"),
        (12, "child"),
        (13, "teen"),
        (15, "teen"),
        (19, "teen"),
        (20, "young adult"),
        (25, "young adult"),
        (35, "young adult"),
        (36, "adult"),
        (45, "adult"),
        (59, "adult"),
        (60, "senior"),
        (70, "senior"),
    ],
)
def test_predict_age_group_maps_ages_and_boundaries(
    age: int, expected_group: str
) -> None:
    assert predict_age_group(age) == expected_group
