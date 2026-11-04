"""Train the transfer-learning age regressor in two phases."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import tensorflow as tf
from tensorflow import keras

from app.models.age_model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "age.h5"
IMAGE_SIZE = (128, 128)
AUTOTUNE = tf.data.AUTOTUNE


def load_labels(data_dir: Path, split: str) -> tuple[list[str], list[float]]:
    """Load image paths and age labels for one prepared dataset split."""
    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. "
            "Run `python3 data/prepare_dataset.py` first."
        )

    labels = pd.read_csv(labels_path)
    required_columns = {"age", "split"}
    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing_columns)}")
    if "filename" not in labels.columns and "source_path" not in labels.columns:
        raise ValueError("labels.csv must contain filename or source_path")

    split_labels = labels[labels["split"] == split]
    if split_labels.empty:
        raise ValueError(f"No records found for split '{split}' in {labels_path}")
    if split_labels["age"].isna().any() or (split_labels["age"] < 0).any():
        raise ValueError(f"Invalid age labels found in split '{split}'")

    path_column = "source_path" if "source_path" in split_labels else "filename"
    paths = [
        str(path) if path_column == "source_path" else str(data_dir / split / path)
        for path in split_labels[path_column]
    ]
    missing_paths = [path for path in paths if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image files listed for '{split}' are missing; "
            f"first missing file: {missing_paths[0]}"
        )
    return paths, split_labels["age"].astype("float32").tolist()


def load_sample_weights(data_dir: Path, split: str) -> list[float]:
    """Load optional confidence weights for exact versus grouped labels."""
    labels = pd.read_csv(data_dir / "labels.csv")
    split_labels = labels[labels["split"] == split]
    if "label_weight" not in split_labels:
        return [1.0] * len(split_labels)
    return split_labels["label_weight"].fillna(1.0).astype("float32").tolist()


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Read and normalize one RGB face image."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(
    paths: list[str],
    labels: list[float],
    sample_weights: list[float],
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """Create a shuffled, batched, prefetched age dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels, sample_weights))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )
    return (
        dataset.map(
            lambda path, label, weight: (decode_image(path, label)[0], label, weight),
            num_parallel_calls=AUTOTUNE,
        )
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )


class ValidationMAELogger(keras.callbacks.Callback):
    """Log validation MAE after each epoch."""

    def __init__(self, phase: str) -> None:
        super().__init__()
        self.phase = phase

    def on_epoch_end(self, epoch: int, logs: dict[str, float] | None = None) -> None:
        logs = logs or {}
        value = logs.get("val_mae")
        if value is not None:
            print(f"{self.phase} epoch {epoch + 1}: validation MAE={value:.4f}")


def make_callbacks(model_path: Path, phase: str) -> list[keras.callbacks.Callback]:
    """Create checkpoint, learning-rate scheduler, and MAE logging callbacks."""
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_mae",
            mode="min",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_mae",
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
        ValidationMAELogger(phase),
    ]


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    head_epochs: int = 10,
    fine_tune_epochs: int = 10,
    batch_size: int = 32,
    fine_tune_learning_rate: float = 1e-5,
    seed: int = 42,
) -> tuple[keras.callbacks.History, keras.callbacks.History]:
    """Train the frozen head, then fine-tune the top 30 backbone layers."""
    if head_epochs <= 0 or fine_tune_epochs <= 0 or batch_size <= 0:
        raise ValueError("epoch counts and batch_size must be positive")
    if fine_tune_learning_rate <= 0:
        raise ValueError("fine_tune_learning_rate must be positive")

    tf.keras.utils.set_random_seed(seed)
    train_paths, train_ages = load_labels(data_dir, "train")
    validation_paths, validation_ages = load_labels(data_dir, "validation")
    train_weights = load_sample_weights(data_dir, "train")
    validation_weights = load_sample_weights(data_dir, "validation")
    train_dataset = make_dataset(
        train_paths, train_ages, train_weights, batch_size, training=True, seed=seed
    )
    validation_dataset = make_dataset(
        validation_paths,
        validation_ages,
        validation_weights,
        batch_size,
        training=False,
        seed=seed,
    )

    model = build_model()
    model_path.parent.mkdir(parents=True, exist_ok=True)

    print("Phase 1: training the regression head with the backbone frozen")
    head_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=head_epochs,
        callbacks=make_callbacks(model_path, "head"),
    )

    backbone = model.get_layer("mobilenetv2_backbone")
    backbone.trainable = True
    for layer in backbone.layers[:-30]:
        layer.trainable = False
    for layer in backbone.layers[-30:]:
        layer.trainable = True

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    print("Phase 2: fine-tuning the top 30 backbone layers")
    fine_tune_history = model.fit(
        train_dataset,
        validation_data=validation_dataset,
        epochs=fine_tune_epochs,
        callbacks=make_callbacks(model_path, "fine-tune"),
    )
    print(f"Best age model saved to {model_path}")
    return head_history, fine_tune_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--head-epochs", type=int, default=10)
    parser.add_argument("--fine-tune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--fine-tune-learning-rate", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    train(**vars(parse_args()))
