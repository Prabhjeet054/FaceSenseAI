import numpy as np
import pytest

from app.services.matcher import FaceMatcher
from app.services.registry import FaceRegistry, RegisteredPerson


def make_registry(embeddings: list[tuple[str, str, np.ndarray]]) -> FaceRegistry:
    registry = FaceRegistry(
        database_url="sqlite:///:memory:",
        detector=object(),
        embedder=object(),
    )
    with registry.session_factory.begin() as session:
        for person_id, name, embedding in embeddings:
            session.add(
                RegisteredPerson(
                    person_id=person_id,
                    name=name,
                    embedding=embedding.astype(np.float32).tobytes(),
                )
            )
    return registry


def test_exact_embedding_match_returns_right_name() -> None:
    alice = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    registry = make_registry([("alice", "Alice", alice)])

    result = FaceMatcher(registry, threshold=0.8).identify(alice)

    assert result["name"] == "Alice"
    assert result["id"] == "alice"
    assert result["score"] == pytest.approx(1.0)


def test_random_embedding_returns_unknown() -> None:
    registered = np.zeros(512, dtype=np.float32)
    registered[0] = 1.0
    query = np.zeros(512, dtype=np.float32)
    query[1] = 1.0
    registry = make_registry([("alice", "Alice", registered)])

    result = FaceMatcher(registry, threshold=0.8).identify(query)

    assert result["name"] == "Unknown"
    assert result["id"] is None


def test_raising_threshold_turns_borderline_match_into_unknown() -> None:
    registered = np.zeros(512, dtype=np.float32)
    registered[:2] = [1.0, 0.0]
    query = np.zeros(512, dtype=np.float32)
    query[:2] = [0.8, 0.6]
    registry = make_registry([("alice", "Alice", registered)])

    borderline = FaceMatcher(registry, threshold=0.7).identify(query)
    rejected = FaceMatcher(registry, threshold=0.9).identify(query)

    assert borderline["name"] == "Alice"
    assert borderline["score"] == pytest.approx(0.8)
    assert rejected["name"] == "Unknown"
