"""Inference adapter for InsightFace's pretrained gender-age model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import onnxruntime as ort
import cv2


class InsightFaceAttributeModel:
    """Expose one output of ``genderage.onnx`` through a Keras-like API."""

    def __init__(self, model_path: str | Path, attribute: str) -> None:
        if attribute not in {"gender", "age"}:
            raise ValueError("attribute must be 'gender' or 'age'")
        self.attribute = attribute
        self.session = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        input_config = self.session.get_inputs()[0]
        self.input_name = input_config.name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, batch: np.ndarray, verbose: int = 0) -> np.ndarray:
        del verbose
        images = np.asarray(batch, dtype=np.float32)
        if images.ndim != 4 or images.shape[-1] != 3:
            raise ValueError("attribute input must have shape (N, H, W, 3)")
        resized = np.stack(
            [cv2.resize(image, (96, 96), interpolation=cv2.INTER_AREA) for image in images]
        )
        nchw = np.transpose((resized * 255.0 - 127.5) / 128.0, (0, 3, 1, 2))
        outputs = np.asarray(
            self.session.run([self.output_name], {self.input_name: nchw})[0]
        )
        if self.attribute == "age":
            return (outputs[:, 2:3] * 100.0).astype(np.float32)
        logits = outputs[:, :2]
        logits -= logits.max(axis=1, keepdims=True)
        probabilities = np.exp(logits)
        probabilities /= probabilities.sum(axis=1, keepdims=True)
        return probabilities[:, 1:2].astype(np.float32)
