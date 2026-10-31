"""Train the gender classifier on the prepared FaceSense AI dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras

from app.models.gender_model import build_model


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "gender.h5"
DEFAULT_PLOT_PATH = PROJECT_ROOT / "models" / "gender_training_curves.png"
IMAGE_SIZE = (128, 128)
AUTOTUNE = tf.data.AUTOTUNE


def load_labels(data_dir: Path, split: str) -> tuple[list[str], list[float]]:
    """Load image paths and gender labels for one prepared dataset split."""
    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(
            f"Labels file not found: {labels_path}. "
            "Run `python3 data/prepare_dataset.py` first."
        )

    labels = pd.read_csv(labels_path)
    required_columns = {"filename", "gender", "split"}
    missing_columns = required_columns - set(labels.columns)
    if missing_columns:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing_columns)}")

    split_labels = labels[labels["split"] == split]
    if split_labels.empty:
        raise ValueError(f"No records found for split '{split}' in {labels_path}")

    paths = [str(data_dir / split / filename) for filename in split_labels["filename"]]
    missing_paths = [path for path in paths if not Path(path).is_file()]
    if missing_paths:
        raise FileNotFoundError(
            f"{len(missing_paths)} image files listed for '{split}' are missing; "
            f"first missing file: {missing_paths[0]}"
        )
    return paths, split_labels["gender"].astype("float32").tolist()


def decode_image(path: tf.Tensor, label: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    """Read and normalize one RGB image."""
    image = tf.io.read_file(path)
    image = tf.io.decode_image(image, channels=3, expand_animations=False)
    image.set_shape((None, None, 3))
    image = tf.image.resize(image, IMAGE_SIZE)
    image = tf.cast(image, tf.float32) / 255.0
    return image, label


def make_dataset(
    paths: list[str],
    labels: list[float],
    batch_size: int,
    training: bool,
    seed: int,
) -> tf.data.Dataset:
    """Create a batched tf.data pipeline for a dataset split."""
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        dataset = dataset.shuffle(
            buffer_size=len(paths), seed=seed, reshuffle_each_iteration=True
        )
    dataset = dataset.map(decode_image, num_parallel_calls=AUTOTUNE)
    return dataset.batch(batch_size).prefetch(AUTOTUNE)


def plot_history(history: keras.callbacks.History, output_path: Path) -> None:
    """Save training and validation accuracy/loss curves."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    epochs = range(1, len(history.history["loss"]) + 1)

    axes[0].plot(epochs, history.history["accuracy"], label="Train")
    axes[0].plot(epochs, history.history["val_accuracy"], label="Validation")
    axes[0].set_title("Gender Classification Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(epochs, history.history["loss"], label="Train")
    axes[1].plot(epochs, history.history["val_loss"], label="Validation")
    axes[1].set_title("Gender Classification Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Binary cross-entropy")
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def train(
    data_dir: Path = DEFAULT_DATA_DIR,
    model_path: Path = DEFAULT_MODEL_PATH,
    plot_path: Path = DEFAULT_PLOT_PATH,
    batch_size: int = 32,
    epochs: int = 30,
    patience: int = 5,
    seed: int = 42,
) -> keras.callbacks.History:
    """Train, checkpoint, and plot the gender model."""
    if batch_size <= 0 or epochs <= 0 or patience < 0:
        raise ValueError("batch_size and epochs must be positive; patience cannot be negative")

    tf.keras.utils.set_random_seed(seed)
    train_paths, train_labels = load_labels(data_dir, "train")
    validation_paths, validation_labels = load_labels(data_dir, "validation")
    train_dataset = make_dataset(
        train_paths, train_labels, batch_size, training=True, seed=seed
    )
    validation_dataset = make_dataset(
        validation_paths, validation_labels, batch_size, training=False, seed=seed
    )

    augmentation = keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.05),
            keras.layers.RandomBrightness(0.10, value_range=(0, 1)),
        ],
        name="gender_augmentation",
    )

    model = build_model()
    augmented_train_dataset = train_dataset.map(
        lambda images, labels: (augmentation(images, training=True), labels),
        num_parallel_calls=AUTOTUNE,
    )
    model_path.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=patience,
            restore_best_weights=True,
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_path,
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
        ),
    ]

    history = model.fit(
        augmented_train_dataset,
        validation_data=validation_dataset,
        epochs=epochs,
        callbacks=callbacks,
    )
    plot_history(history, plot_path)
    print(f"Best model saved to {model_path}")
    print(f"Training curves saved to {plot_path}")
    return history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--plot-path", type=Path, default=DEFAULT_PLOT_PATH)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    train(**vars(arguments))
