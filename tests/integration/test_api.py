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
                "emotion": "happy",
                "emotion_confidence": 0.91,
            }
        ]


class FakeRegistry:
    def __init__(self) -> None:
        self.people = {"person-1": {"id": "person-1", "name": "Rahul"}}

    def add(
        self,
        name: str,
        person_id: str,
        photos: list[np.ndarray],
        **kwargs: object,
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


class SpoofLiveness:
    def check(self, frames: list[np.ndarray]) -> dict[str, object]:
        return {"live": False, "status": "spoof", "score": 0.1}


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    services = main.Services(registry=FakeRegistry(), pipeline=FakePipeline())
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)
    return TestClient(main.app, headers={"X-API-Key": "test-api-key"})


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
        data={"name": "Priya", "person_id": "person-2", "consent": "true"},
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


def test_api_requires_valid_api_key(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing_key = TestClient(main.app)
    assert missing_key.get("/health").status_code == 401

    monkeypatch.setenv("FACESENSE_API_KEY", "changed-key")
    assert client.get("/health").status_code == 401


def test_registration_requires_consent(client: TestClient) -> None:
    files = [
        ("photos", ("face-1.jpg", image_bytes(), "image/jpeg")),
        ("photos", ("face-2.jpg", image_bytes(), "image/jpeg")),
        ("photos", ("face-3.jpg", image_bytes(), "image/jpeg")),
    ]
    response = client.post(
        "/register",
        data={"name": "Priya", "person_id": "person-2", "consent": "false"},
        files=files,
    )
    assert response.status_code == 422
    assert response.json()["detail"] == "Consent is required for registration."


def test_oversized_image_upload_returns_payload_too_large(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    limit = 1024 * 1024
    monkeypatch.setattr(config, "API_MAX_IMAGE_SIZE_BYTES", limit)
    response = client.post("/analyze", files=[upload(content=b"x" * (limit + 1))])

    assert response.status_code == 413
    assert response.json()["detail"] == "Image exceeds the 1 MB limit."


def test_registration_and_attendance_block_spoof_attempts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    services = main.Services(
        registry=FakeRegistry(),
        pipeline=FakePipeline(),
        liveness=SpoofLiveness(),
    )
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)
    client = TestClient(main.app, headers={"X-API-Key": "test-api-key"})
    photos = [
        ("photos", (f"face-{index}.jpg", image_bytes(), "image/jpeg"))
        for index in range(3)
    ]

    registration = client.post(
        "/register",
        data={"name": "Priya", "person_id": "person-2", "consent": "true"},
        files=photos,
    )
    attendance = client.post(
        "/attendance",
        files=[
            ("frames", (f"frame-{index}.jpg", image_bytes(), "image/jpeg"))
            for index in range(2)
        ],
    )

    assert registration.status_code == 422
    assert "spoof detected" in registration.json()["detail"]
    assert attendance.status_code == 422
    assert "spoof detected" in attendance.json()["detail"]


def test_logs_and_stats_endpoints_support_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    services = main.Services(
        registry=FakeRegistry(),
        pipeline=FakePipeline(),
        logs=RecognitionLogService(database_url="sqlite:///:memory:"),
    )
    monkeypatch.setattr(main, "_services", services)
    monkeypatch.setattr(main, "_startup_error", None)
    client = TestClient(main.app, headers={"X-API-Key": "test-api-key"})

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


def test_invalid_image_bytes_return_bad_request(client: TestClient) -> None:
    response = client.post(
        "/analyze",
        files=[("image", ("face.jpg", b"not-image", "image/jpeg"))],
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Uploaded file is not a valid image."


def test_attendance_success_marks_present(monkeypatch: pytest.MonkeyPatch) -> None:
    class LiveLiveness:
        def check(self, frames: list[np.ndarray]) -> dict[str, object]:
            return {"live": True, "status": "live", "score": 0.9}

    class Attendance:
        def __init__(self) -> None:
            self.people: list[str] = []

        def mark_present(self, person_id: str, source: str) -> dict[str, object]:
            self.people.append(person_id)
            return {"marked": True}

    attendance_service = Attendance()
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    monkeypatch.setattr(
        main,
        "_services",
        main.Services(
            registry=FakeRegistry(),
            pipeline=FakePipeline(),
            liveness=LiveLiveness(),
            attendance=attendance_service,
        ),
    )
    monkeypatch.setattr(main, "_startup_error", None)
    client = TestClient(main.app, headers={"X-API-Key": "test-api-key"})

    response = client.post(
        "/attendance",
        files=[
            ("frames", (f"frame-{index}.jpg", image_bytes(), "image/jpeg"))
            for index in range(2)
        ],
    )

    assert response.status_code == 200
    assert attendance_service.people == ["person-1"]


def test_service_startup_fails_closed_without_encryption_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("FACESENSE_API_KEY", "test-api-key")
    monkeypatch.delenv("FACESENSE_ENCRYPTION_KEY", raising=False)
    monkeypatch.setattr(main, "_services", None)
    monkeypatch.setattr(main, "_startup_error", None)

    response = TestClient(main.app, headers={"X-API-Key": "test-api-key"}).get("/logs")

    assert response.status_code == 503
    assert "FACESENSE_ENCRYPTION_KEY" in response.json()["detail"]
