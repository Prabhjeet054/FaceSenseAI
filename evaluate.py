"""Evaluate the trained gender and age models on the held-out test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"
DEFAULT_GENDER_MODEL = PROJECT_ROOT / "models" / "gender.h5"
DEFAULT_AGE_MODEL = PROJECT_ROOT / "models" / "age.h5"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "metrics.json"
IMAGE_SIZE = (128, 128)
AGE_BUCKETS = {
    "child": (0, 12),
    "teen": (13, 19),
    "young adult": (20, 35),
    "adult": (36, 59),
    "senior": (60, float("inf")),
}


def load_test_data(data_dir: Path) -> tuple[np.ndarray, pd.DataFrame]:
    """Load test images and labels in the same order."""
    labels_path = data_dir / "labels.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    labels = pd.read_csv(labels_path)
    required = {"filename", "age", "gender", "split"}
    missing = required - set(labels.columns)
    if missing:
        raise ValueError(f"labels.csv is missing columns: {sorted(missing)}")
    test_labels = labels[labels["split"] == "test"].reset_index(drop=True)
    if test_labels.empty:
        raise ValueError("No test records found in labels.csv")

    images: list[np.ndarray] = []
    for filename in test_labels["filename"]:
        image_path = data_dir / "test" / filename
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image not found: {image_path}")
        image = tf.io.decode_image(
            tf.io.read_file(str(image_path)),
            channels=3,
            expand_animations=False,
        )
        image = tf.image.resize(image, IMAGE_SIZE)
        images.append((tf.cast(image, tf.float32) / 255.0).numpy())
    return np.stack(images), test_labels


def bucket_for_age(age: float) -> str:
    for bucket, (minimum, maximum) in AGE_BUCKETS.items():
        if minimum <= age <= maximum:
            return bucket
    raise ValueError(f"Age is outside supported buckets: {age}")


def plot_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=[0, 1],
        yticks=[0, 1],
        xticklabels=["Male", "Female"],
        yticklabels=["Male", "Female"],
        xlabel="Predicted",
        ylabel="Actual",
        title="Gender Confusion Matrix",
    )
    for row in range(2):
        for column in range(2):
            axis.text(column, row, matrix[row, column], ha="center", va="center")
    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_age_errors(errors: dict[str, float], output_path: Path) -> None:
    figure, axis = plt.subplots(figsize=(8, 4))
    axis.bar(errors.keys(), errors.values(), color="steelblue")
    axis.set_title("Mean Absolute Age Error by Age Bucket")
    axis.set_xlabel("True age bucket")
    axis.set_ylabel("MAE (years)")
    axis.tick_params(axis="x", rotation=20)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def evaluate(
    data_dir: Path = DEFAULT_DATA_DIR,
    gender_model_path: Path = DEFAULT_GENDER_MODEL,
    age_model_path: Path = DEFAULT_AGE_MODEL,
    report_path: Path = DEFAULT_REPORT_PATH,
) -> dict[str, object]:
    """Evaluate both models, save metrics JSON, and save PNG plots."""
    for model_path in (gender_model_path, age_model_path):
        if not model_path.is_file():
            raise FileNotFoundError(f"Model file not found: {model_path}")

    images, labels = load_test_data(data_dir)
    gender_model = tf.keras.models.load_model(gender_model_path, compile=False)
    age_model = tf.keras.models.load_model(age_model_path, compile=False)
    gender_scores = np.asarray(gender_model.predict(images, verbose=0)).reshape(-1)
    age_predictions = np.asarray(age_model.predict(images, verbose=0)).reshape(-1)
    gender_actual = labels["gender"].to_numpy(dtype=int)
    gender_predicted = (gender_scores >= 0.5).astype(int)
    age_actual = labels["age"].to_numpy(dtype=float)

    matrix = confusion_matrix(gender_actual, gender_predicted, labels=[0, 1])
    age_errors = np.abs(age_predictions - age_actual)
    bucket_errors: dict[str, float] = {}
    for bucket in AGE_BUCKETS:
        selected = np.array([bucket_for_age(age) == bucket for age in age_actual])
        bucket_errors[bucket] = (
            float(age_errors[selected].mean()) if selected.any() else None
        )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    confusion_path = report_path.parent / "gender_confusion_matrix.png"
    age_error_path = report_path.parent / "age_error_by_bucket.png"
    plot_confusion_matrix(matrix, confusion_path)
    plot_age_errors(
        {bucket: value or 0.0 for bucket, value in bucket_errors.items()},
        age_error_path,
    )
    metrics: dict[str, object] = {
        "samples": int(len(labels)),
        "gender_accuracy": float(accuracy_score(gender_actual, gender_predicted)),
        "age_mae": float(age_errors.mean()),
        "gender": {
            "accuracy": float(accuracy_score(gender_actual, gender_predicted)),
            "precision": float(
                precision_score(gender_actual, gender_predicted, zero_division=0)
            ),
            "recall": float(
                recall_score(gender_actual, gender_predicted, zero_division=0)
            ),
            "f1": float(f1_score(gender_actual, gender_predicted, zero_division=0)),
            "confusion_matrix": matrix.tolist(),
        },
        "age": {
            "mae": float(age_errors.mean()),
            "error_by_age_bucket": bucket_errors,
        },
        "plots": {
            "gender_confusion_matrix": str(confusion_path),
            "age_error_by_bucket": str(age_error_path),
        },
    }
    report_path.write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--gender-model", type=Path, default=DEFAULT_GENDER_MODEL)
    parser.add_argument("--age-model", type=Path, default=DEFAULT_AGE_MODEL)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(**vars(parse_args()))
