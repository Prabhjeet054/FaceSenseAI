import math

import pytest

tf = pytest.importorskip("tensorflow")
import pandas as pd

from train_gender import train


def test_gender_training_smoke_saves_model_and_finite_loss(tmp_path) -> None:
    data_dir = tmp_path / "processed"
    train_dir = data_dir / "train"
    validation_dir = data_dir / "validation"
    train_dir.mkdir(parents=True)
    validation_dir.mkdir(parents=True)

    records = []
    for index in range(32):
        split = "train" if index < 24 else "validation"
        filename = f"sample_{index:02d}.png"
        image = tf.random.uniform(
            shape=(128, 128, 3),
            minval=0,
            maxval=256,
            dtype=tf.int32,
            seed=index,
        )
        image_bytes = tf.io.encode_png(tf.cast(image, tf.uint8))
        tf.io.write_file(str(data_dir / split / filename), image_bytes)
        records.append(
            {
                "filename": filename,
                "age": 20 + index,
                "gender": index % 2,
                "split": split,
            }
        )

    pd.DataFrame(records).to_csv(data_dir / "labels.csv", index=False)
    model_path = tmp_path / "models" / "gender.h5"

    history = train(
        data_dir=data_dir,
        model_path=model_path,
        plot_path=tmp_path / "plots" / "training_curves.png",
        batch_size=8,
        epochs=1,
        patience=1,
        seed=42,
    )

    assert model_path.is_file()
    assert history.history["loss"]
    assert math.isfinite(history.history["loss"][-1])
