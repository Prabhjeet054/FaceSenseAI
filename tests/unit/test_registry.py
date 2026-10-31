import numpy as np
import pytest

from app.services.registry import FaceRegistry


class FakeDetector:
    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
        if image[0, 0, 0] == 0:
            return []
        return [
            {
                "bbox": [10, 10, 80, 80],
                "confidence": 0.99,
                "landmarks": [
                    [30, 30],
                    [70, 30],
                    [50, 50],
                    [35, 70],
                    [65, 70],
                ],
            }
        ]


class FakeEmbedder:
    def embed(self, face: np.ndarray) -> np.ndarray:
        return np.ones(512, dtype=np.float32)


@pytest.fixture
def registry() -> FaceRegistry:
    return FaceRegistry(
        database_url="sqlite:///:memory:",
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
    )


@pytest.fixture
def valid_photos() -> list[np.ndarray]:
    return [np.full((100, 100, 3), 255, dtype=np.uint8) for _ in range(3)]


def test_register_and_list_person(
    registry: FaceRegistry, valid_photos: list[np.ndarray]
) -> None:
    person = registry.add("Alice", "person-001", valid_photos)

    assert person["id"] == "person-001"
    assert person["name"] == "Alice"
    assert len(person["embedding"]) == 512
    assert registry.list()[0]["name"] == "Alice"


def test_update_person_name(
    registry: FaceRegistry, valid_photos: list[np.ndarray]
) -> None:
    registry.add("Alice", "person-001", valid_photos)

    updated = registry.update("person-001", name="Alice Updated")

    assert updated["name"] == "Alice Updated"
    assert registry.list()[0]["name"] == "Alice Updated"


def test_delete_person(
    registry: FaceRegistry, valid_photos: list[np.ndarray]
) -> None:
    registry.add("Alice", "person-001", valid_photos)

    registry.delete("person-001")

    assert registry.list() == []


def test_rejects_photo_with_no_face(
    registry: FaceRegistry, valid_photos: list[np.ndarray]
) -> None:
    no_face = np.zeros((100, 100, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="contains no face"):
        registry.add("Alice", "person-001", [valid_photos[0], no_face, valid_photos[2]])
