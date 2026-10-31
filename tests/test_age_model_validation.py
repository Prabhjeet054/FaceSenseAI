from pathlib import Path

import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

import config


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
MODEL_PATH = PROJECT_ROOT / "models" / "age.h5"


def test_saved_age_model_mae_on_validation_samples() -> None:
    if not MODEL_PATH.is_file():
        pytest.skip(f"Saved age model not found: {MODEL_PATH}")

    labels_path = DATA_DIR / "labels.csv"
    if not labels_path.is_file():
        pytest.skip(f"Validation labels not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    required_columns = {"filename", "age", "split"}
    missing_columns = required_columns - set(labels.columns)
    assert not missing_columns, f"labels.csv is missing columns: {sorted(missing_columns)}"

    validation_labels = labels[labels["split"] == "validation"].head(10)
    if validation_labels.empty:
        pytest.skip("No validation images are available")

    image_paths = [
        DATA_DIR / "validation" / filename
        for filename in validation_labels["filename"]
    ]
    missing_images = [path for path in image_paths if not path.is_file()]
    if missing_images:
        pytest.skip(f"Validation image not found: {missing_images[0]}")

    images = tf.stack(
        [
            tf.image.resize(
                tf.cast(
                    tf.io.decode_image(
                        tf.io.read_file(str(path)),
                        channels=3,
                        expand_animations=False,
                    ),
                    tf.float32,
                )
                / 255.0,
                (128, 128),
            )
            for path in image_paths
        ]
    )
    expected_ages = tf.constant(validation_labels["age"].to_numpy(), dtype=tf.float32)
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    predictions = tf.squeeze(model.predict(images, verbose=0), axis=-1)
    mae = float(tf.reduce_mean(tf.abs(predictions - expected_ages)).numpy())

    assert mae < config.AGE_VALIDATION_MAE_LIMIT, (
        f"Validation MAE {mae:.2f} exceeds configured threshold "
        f"{config.AGE_VALIDATION_MAE_LIMIT:.2f}"
    )
