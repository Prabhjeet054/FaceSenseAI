from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.api.main import app


@pytest.fixture
def sample_image() -> np.ndarray:
    """Return a small deterministic BGR image for image-processing tests."""
    return np.full((120, 120, 3), 127, dtype=np.uint8)


@pytest.fixture
def sample_images(sample_image: np.ndarray) -> list[np.ndarray]:
    """Return a reusable set of deterministic sample images."""
    return [sample_image.copy() for _ in range(3)]


@pytest.fixture
def sample_image_bytes(sample_image: np.ndarray) -> bytes:
    """Encode the shared sample image as JPEG upload content."""
    success, encoded = cv2.imencode(".jpg", sample_image)
    if not success:
        raise RuntimeError("Could not encode the sample image")
    return encoded.tobytes()


@pytest.fixture
def temp_database(tmp_path: Path) -> str:
    """Return a temporary SQLite URL for service integration tests."""
    return f"sqlite:///{tmp_path / 'facesense-test.db'}"


@pytest.fixture
def test_client(
    monkeypatch: pytest.MonkeyPatch,
) -> TestClient:
    """Return an authenticated client for API smoke and integration tests."""
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    return TestClient(app, headers={"X-API-Key": "test-api-key"})
