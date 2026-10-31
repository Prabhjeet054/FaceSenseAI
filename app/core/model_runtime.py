"""Inference adapters for converted TensorFlow Lite models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class TFLiteModel:
    """Expose a TensorFlow Lite model through the Keras-like ``predict`` API."""

    def __init__(self, model_path: str | Path) -> None:
        try:
            import tensorflow as tf
        except ImportError as error:
            raise ImportError("TensorFlow is required to run TFLite models") from error

        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        if len(self.input_details) != 1 or len(self.output_details) != 1:
            raise ValueError("TFLite model must have exactly one input and one output")

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        inputs = np.asarray(batch, dtype=np.float32)
        input_detail = self.input_details[0]
        shape = tuple(int(value) for value in inputs.shape)
        expected_shape = tuple(int(value) for value in input_detail["shape"])
        if len(shape) != len(expected_shape) or shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"Input shape {shape} does not match TFLite model shape {expected_shape}"
            )
        if expected_shape[0] != shape[0]:
            self.interpreter.resize_tensor_input(input_detail["index"], shape)
            self.interpreter.allocate_tensors()
            input_detail = self.interpreter.get_input_details()[0]
        self.interpreter.set_tensor(input_detail["index"], inputs)
        self.interpreter.invoke()
        return np.asarray(
            self.interpreter.get_tensor(self.interpreter.get_output_details()[0]["index"])
        )


__all__ = ["TFLiteModel"]
