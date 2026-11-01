"""Face embeddings with InsightFace ArcFace."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


EMBEDDING_DIMENSION = 512
DEFAULT_MODEL_NAME = "w600k_r50.onnx"
DEFAULT_MODEL_PATH = (
    Path.home() / ".insightface" / "models" / DEFAULT_MODEL_NAME
)


class FaceEmbedder:
    """Create L2-normalized 512-dimensional ArcFace embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        providers: Sequence[str] | None = None,
        model: Any | None = None,
    ) -> None:
        if model is not None:
            self.model = model
            return

        try:
            from insightface.model_zoo import get_model
        except ImportError as error:
            raise ImportError(
                "InsightFace is required to create FaceEmbedder. "
                "Install it with `python3 -m pip install insightface onnxruntime`."
            ) from error

        model_name_or_path = (
            str(DEFAULT_MODEL_PATH)
            if model_name == DEFAULT_MODEL_NAME and DEFAULT_MODEL_PATH.is_file()
            else model_name
        )
        kwargs: dict[str, Any] = {}
        if providers is not None:
            kwargs["providers"] = list(providers)
        try:
            self.model = get_model(model_name_or_path, **kwargs)
            if self.model is None:
                raise RuntimeError(
                    f"InsightFace model '{model_name_or_path}' was not found"
                )
            self.model.prepare(ctx_id=0)
        except Exception as error:
            raise RuntimeError(
                f"Could not load InsightFace model '{model_name}'. "
                "Download the InsightFace buffalo_l model pack and place "
                f"{DEFAULT_MODEL_NAME} at {DEFAULT_MODEL_PATH}."
            ) from error

    def embed(
        self, faces: np.ndarray | Sequence[np.ndarray]
    ) -> np.ndarray:
        """Return one ``(512,)`` or a batch of ``(N, 512)`` normalized embeddings."""
        batch, single_input = self._as_batch(faces)
        try:
            embeddings = np.asarray(self.model.get_feat(batch), dtype=np.float32)
        except Exception as error:
            raise RuntimeError("InsightFace failed to create face embeddings") from error

        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)
        if embeddings.shape != (len(batch), EMBEDDING_DIMENSION):
            raise ValueError(
                "InsightFace returned embeddings with unexpected shape: "
                f"{embeddings.shape}; expected ({len(batch)}, {EMBEDDING_DIMENSION})"
            )

        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        if np.any(norms == 0) or not np.isfinite(norms).all():
            raise ValueError("InsightFace returned a zero or non-finite embedding")
        normalized = embeddings / norms
        return normalized[0] if single_input else normalized

    @staticmethod
    def _as_batch(
        faces: np.ndarray | Sequence[np.ndarray],
    ) -> tuple[list[np.ndarray], bool]:
        if faces is None:
            raise ValueError("No face was passed; provide an aligned face image")

        if isinstance(faces, np.ndarray):
            if faces.size == 0:
                raise ValueError("No face was passed; the face image is empty")
            if faces.ndim == 3:
                batch = [faces]
                single_input = True
            elif faces.ndim == 4:
                batch = list(faces)
                single_input = False
            else:
                raise ValueError("A face must have shape (H, W, C) or (N, H, W, C)")
        else:
            batch = list(faces)
            single_input = False
            if not batch:
                raise ValueError("No face was passed; the face batch is empty")

        for index, face in enumerate(batch):
            if not isinstance(face, np.ndarray) or face.size == 0:
                raise ValueError(f"Face at index {index} is empty or invalid")
            if face.ndim != 3 or face.shape[2] not in (3, 4):
                raise ValueError(
                    f"Face at index {index} must have shape (H, W, 3) or (H, W, 4)"
                )
        return batch, single_input


__all__ = ["DEFAULT_MODEL_NAME", "EMBEDDING_DIMENSION", "FaceEmbedder"]
