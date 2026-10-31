"""Convert trained Keras models to TensorFlow Lite for inference."""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf


def convert_model(source: Path, destination: Path) -> None:
    """Convert one Keras model and write a float32 TFLite artifact."""
    if not source.is_file():
        raise FileNotFoundError(f"Model file not found: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(source, compile=False)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    destination.write_bytes(converter.convert())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gender-model", type=Path, default=Path("models/gender.h5"))
    parser.add_argument("--age-model", type=Path, default=Path("models/age.h5"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/tflite"))
    args = parser.parse_args()
    convert_model(args.gender_model, args.output_dir / "gender.tflite")
    convert_model(args.age_model, args.output_dir / "age.tflite")
    print(f"Converted models written to {args.output_dir}")


if __name__ == "__main__":
    main()
