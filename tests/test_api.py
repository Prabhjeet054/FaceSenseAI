from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

import config
from app.api import main
from app.services.logs import RecognitionLogService


class FakePipeline:
    def analyze(self, image: np.ndarray) -> list[dict[str, Any]]:
        assert image.ndim == 3
        return [
            {
                "box": [1.0, 2.0, 30.0, 40.0],
                "name": "Rahul",
                "person_id": "person-1",
                "match_score": 0.94,
                "gender": "male",
                "gender_confidence": 0.94,
                "age": 27.0,
                "age_group": "young adult",
            }
        ]


class FakeRegistry:
    def __init__(self) -> None:
        self.people = {"person-1": {"id": "person-1", "name": "Rahul"}}

    def add(
        self, name: str, person_id: str, photos: list[np.ndarray]
    ) -> dict[str, str]:
        assert len(photos) == 3
        person = {"id": person_id, "name": name}
        self.people[person_id] = person
        return person

    def list(self) -> list[dict[str, str]]:
        return list(self.people.values())

    def delete(self, person_id: str) -> None:
        if person_id not in self.people:
            raise KeyError(f"Person ID is not registered: {person_id}")
        del self.people[person_id]


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    services = main.Services(registry=FakeRegistry(), pipeline=FakePipeline())
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)
    return TestClient(main.app)


def image_bytes() -> bytes:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success
    return encoded.tobytes()


def upload(
    name: str = "face.jpg", content: bytes | None = None
) -> tuple[str, tuple[str, bytes, str]]:
    return ("image", (name, image_bytes() if content is None else content, "image/jpeg"))


def test_api_endpoints_and_upload_validation(client: TestClient) -> None:
    health_response = client.get("/health")
    assert health_response.status_code == 200
    assert health_response.json() == {"status": "ok", "models_loaded": True}

    analyze_response = client.post("/analyze", files=[upload()])
    assert analyze_response.status_code == 200
    assert analyze_response.json()["faces"][0]["name"] == "Rahul"

    register_files = [
        ("photos", ("face-1.jpg", image_bytes(), "image/jpeg")),
        ("photos", ("face-2.jpg", image_bytes(), "image/jpeg")),
        ("photos", ("face-3.jpg", image_bytes(), "image/jpeg")),
    ]
    register_response = client.post(
        "/register",
        data={"name": "Priya", "person_id": "person-2"},
        files=register_files,
    )
    assert register_response.status_code == 201
    assert register_response.json() == {"id": "person-2", "name": "Priya"}

    people_response = client.get("/people")
    assert people_response.status_code == 200
    assert {person["id"] for person in people_response.json()} == {
        "person-1",
        "person-2",
    }

    delete_response = client.delete("/people/person-2")
    assert delete_response.status_code == 204
    assert delete_response.content == b""


def test_text_file_upload_returns_bad_request(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        files=[("image", ("notes.txt", b"not an image", "text/plain"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Unsupported image type. Upload a JPEG, PNG, or WebP image."
    )


def test_oversized_image_upload_returns_payload_too_large(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = 1024 * 1024
    monkeypatch.setattr(config, "API_MAX_IMAGE_SIZE_BYTES", limit)
    response = client.post("/analyze", files=[upload(content=b"x" * (limit + 1))])

    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds the 1 MB limit."


def test_logs_and_stats_endpoints_support_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    services = main.Services(
        registry=FakeRegistry(),
        pipeline=FakePipeline(),
        logs=RecognitionLogService(database_url="sqlite:///:memory:"),
    )
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)
    client = TestClient(main.app)

    analyze_response = client.post("/analyze", files=[upload()])
    assert analyze_response.status_code == 200

    logs_response = client.get("/logs", params={"person_id": "person-1"})
    assert logs_response.status_code == 200
    logs = logs_response.json()
    assert len(logs) == 1
    assert logs[0]["person_id"] == "person-1"
    assert logs[0]["source"] == "api"

    stats_response = client.get("/stats", params={"person_id": "person-1"})
    assert stats_response.status_code == 200
    assert stats_response.json()["total"] == 1
    assert stats_response.json()["by_gender"] == {"male": 1}
    assert stats_response.json()["by_age_group"] == {"young adult": 1}
