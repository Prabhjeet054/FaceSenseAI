from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.orm import Session

from app.api import main
from app.services.logs import RecognitionLogService
from app.services.registry import FaceRegistry, RegisteredPerson


class FakeDetector:
    def detect(self, image: np.ndarray) -> list[dict[str, object]]:
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


class FakePipeline:
    def analyze(self, image: np.ndarray) -> list[dict[str, Any]]:
        return []


class FakeRegistry(FaceRegistry):
    def __init__(self) -> None:
        pass

    def add(
        self,
        name: str,
        person_id: str,
        photos: list[np.ndarray],
        **kwargs: object,
    ) -> dict[str, str]:
        raise AssertionError("registration should stop before persistence")


def image_bytes() -> bytes:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def registration_files() -> list[tuple[str, tuple[str, bytes, str]]]:
    return [
        ("photos", (f"face-{index}.jpg", image_bytes(), "image/jpeg"))
        for index in range(3)
    ]


def test_registration_without_consent_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    services = main.Services(
        registry=FakeRegistry(),
        pipeline=FakePipeline(),
    )
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)

    client = TestClient(main.app, headers={"X-API-Key": "test-api-key"})
    response = client.post(
        "/register",
        data={"name": "Alice", "person_id": "alice", "consent": "false"},
        files=registration_files(),
    )

    assert response.status_code == 422
    assert response.json()["detail"] == "Consent is required for registration."


def test_request_without_api_key_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")

    response = TestClient(main.app).get("/health")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or missing API key"


def test_stored_embedding_is_encrypted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cryptography.fernet import Fernet

    monkeypatch.setenv("FACESENSE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    registry = FaceRegistry(
        database_url=f"sqlite:///{tmp_path / 'registry.db'}",
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
    )
    raw_embedding = np.ones(512, dtype=np.float32).tobytes()

    registry.add("Alice", "alice", [np.ones((100, 100, 3), dtype=np.uint8)] * 3)

    with Session(registry.engine) as session:
        record = session.get(RegisteredPerson, "alice")

    assert record is not None
    assert record.embedding != raw_embedding
    assert not np.array_equal(
        np.frombuffer(record.embedding[: 512], dtype=np.uint8),
        np.frombuffer(raw_embedding[: 512], dtype=np.uint8),
    )


def test_delete_my_data_removes_embedding_and_logs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from cryptography.fernet import Fernet

    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    monkeypatch.setenv("FACESENSE_ENCRYPTION_KEY", Fernet.generate_key().decode())
    registry = FaceRegistry(
        database_url=f"sqlite:///{tmp_path / 'registry.db'}",
        detector=FakeDetector(),
        embedder=FakeEmbedder(),
    )
    logs = RecognitionLogService(engine=registry.engine)
    registry.add("Alice", "alice", [np.ones((100, 100, 3), dtype=np.uint8)] * 3)
    logs.log(
        person_id="alice",
        event_time=datetime.now(timezone.utc),
        gender="female",
        age=30,
        confidence=0.95,
        source="test",
    )
    services = main.Services(
        registry=registry,
        pipeline=FakePipeline(),
        logs=logs,
    )
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)

    response = TestClient(main.app, headers={"X-API-Key": "test-api-key"}).delete(
        "/people/alice/data"
    )

    assert response.status_code == 200
    assert response.json()["deleted_logs"] == 1
    assert registry.list() == []
    assert logs.list(person_id="alice") == []
    with Session(registry.engine) as session:
        assert session.get(RegisteredPerson, "alice") is None
