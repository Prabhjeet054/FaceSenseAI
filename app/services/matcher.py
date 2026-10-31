"""Identify faces by cosine similarity against the registry."""

from __future__ import annotations

from typing import Any

import numpy as np
from sqlalchemy import select
from sqlalchemy.orm import Session

import config
from app.core.embedder import EMBEDDING_DIMENSION
from app.services.registry import FaceRegistry, RegisteredPerson


class FaceMatcher:
    """Match query embeddings against registered people."""

    def __init__(
        self,
        registry: FaceRegistry,
        threshold: float = config.FACE_RECOGNITION_THRESHOLD,
    ) -> None:
        if not 0 <= threshold <= 1:
            raise ValueError("threshold must be between 0 and 1")
        self.registry = registry
        self.threshold = threshold

    def identify(
        self, embedding: np.ndarray, top_k: int = 1
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return the best match or ranked candidates.

        With the default ``top_k=1``, the result is a dictionary containing
        ``name``, ``id``, and ``score``. If the best score is below the
        threshold, ``name`` is ``"Unknown"`` and ``id`` is ``None``.
        Set ``top_k=3`` to receive the three highest-scoring candidates,
        regardless of threshold, in descending score order.
        """
        query = self._normalize_query(embedding)
        if top_k <= 0:
            raise ValueError("top_k must be a positive integer")

        with Session(self.registry.engine) as session:
            records = session.scalars(
                select(RegisteredPerson).order_by(RegisteredPerson.person_id)
            ).all()

        candidates = sorted(
            (self._candidate(record, query) for record in records),
            key=lambda candidate: candidate["score"],
            reverse=True,
        )
        candidates = candidates[:top_k]
        if top_k > 1:
            return candidates
        if not candidates or candidates[0]["score"] < self.threshold:
            score = candidates[0]["score"] if candidates else 0.0
            return {"name": "Unknown", "id": None, "score": score}
        return candidates[0]

    @staticmethod
    def _normalize_query(embedding: np.ndarray) -> np.ndarray:
        query = np.asarray(embedding, dtype=np.float32)
        if query.shape != (EMBEDDING_DIMENSION,):
            raise ValueError(
                f"embedding must have shape ({EMBEDDING_DIMENSION},), got {query.shape}"
            )
        norm = np.linalg.norm(query)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("embedding must have a non-zero finite norm")
        return query / norm

    def _candidate(self, record: RegisteredPerson, query: np.ndarray) -> dict[str, Any]:
        stored = self.registry.decrypt_embedding(record.embedding)
        norm = np.linalg.norm(stored)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError(f"Stored embedding for {record.person_id} is invalid")
        score = float(np.dot(query, stored / norm))
        return {"name": record.name, "id": record.person_id, "score": score}


__all__ = ["FaceMatcher"]
