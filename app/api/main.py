"""FastAPI application for FaceSense AI."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Annotated, Any

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict

import config
from app.core.embedder import FaceEmbedder
from app.core.liveness import LivenessChecker
from app.services.matcher import FaceMatcher
from app.services.logs import RecognitionLogService
from app.services.pipeline import FacePipeline
from app.services.registry import FaceRegistry


class FaceResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    box: list[float]
    name: str
    person_id: str | None = None
    match_score: float
    gender: str
    gender_confidence: float
    age: float
    age_group: str
    emotion: str
    emotion_confidence: float


class AnalyzeResponse(BaseModel):
    faces: list[FaceResult]


class PersonResponse(BaseModel):
    id: str
    name: str


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool


class RecognitionLogResponse(BaseModel):
    id: int
    person_id: str | None
    time: str
    gender: str
    age: float
    age_group: str
    confidence: float
    source: str


class StatsResponse(BaseModel):
    total: int
    by_gender: dict[str, int]
    by_age_group: dict[str, int]
    by_day: dict[str, int]


@dataclass
class Services:
    registry: FaceRegistry
    pipeline: FacePipeline
    logs: RecognitionLogService | Any | None = None
    liveness: LivenessChecker | Any | None = None


app = FastAPI(title="FaceSense AI API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
_services: Services | None = None
_startup_error: str | None = None


def get_services() -> Services:
    """Load shared models and services once on first model-dependent request."""
    global _services, _startup_error
    if _services is not None:
        return _services
    if _startup_error is not None:
        raise HTTPException(status_code=503, detail=_startup_error)
    try:
        embedder = FaceEmbedder()
        registry = FaceRegistry(embedder=embedder)
        pipeline = FacePipeline(
            detector=registry.detector,
            embedder=embedder,
            matcher=FaceMatcher(registry),
        )
        _services = Services(
            registry=registry,
            pipeline=pipeline,
            logs=RecognitionLogService(engine=registry.engine),
            liveness=LivenessChecker(),
        )
        return _services
    except (FileNotFoundError, ImportError, RuntimeError) as error:
        _startup_error = f"FaceSense AI services are unavailable: {error}"
        raise HTTPException(status_code=503, detail=_startup_error) from error


async def read_image_upload(upload: UploadFile) -> np.ndarray:
    """Validate and decode an uploaded image."""
    content_type = upload.content_type or ""
    suffix = Path(upload.filename or "").suffix.lower()
    if content_type not in config.API_ALLOWED_IMAGE_TYPES or suffix not in {
        ".jpg",
        ".jpeg",
        ".png",
        ".webp",
    }:
        raise HTTPException(
            status_code=400,
            detail="Unsupported image type. Upload a JPEG, PNG, or WebP image.",
        )
    data = await upload.read(config.API_MAX_IMAGE_SIZE_BYTES + 1)
    if len(data) > config.API_MAX_IMAGE_SIZE_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"Image exceeds the {config.API_MAX_IMAGE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.")
    return image


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(
    image: Annotated[UploadFile, File(description="JPEG, PNG, or WebP image")],
) -> AnalyzeResponse:
    services = get_services()
    image_array = await read_image_upload(image)
    try:
        results = services.pipeline.analyze(image_array)
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=f"Could not analyze image: {error}") from error
    if services.logs is not None:
        for result in results:
            services.logs.log(
                person_id=result.get("person_id"),
                event_time=None,
                gender=result["gender"],
                age=result["age"],
                confidence=result["match_score"],
                source="api",
            )
    return AnalyzeResponse(faces=results)


@app.get("/logs", response_model=list[RecognitionLogResponse])
def list_logs(
    date_from: date | None = None,
    date_to: date | None = None,
    person_id: str | None = None,
) -> list[RecognitionLogResponse]:
    services = get_services()
    try:
        return services.logs.list(
            date_from=date_from, date_to=date_to, person_id=person_id
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/stats", response_model=StatsResponse)
def stats(
    date_from: date | None = None,
    date_to: date | None = None,
    person_id: str | None = None,
) -> StatsResponse:
    services = get_services()
    try:
        return services.logs.stats(
            date_from=date_from, date_to=date_to, person_id=person_id
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/register", response_model=PersonResponse, status_code=201)
async def register(
    name: Annotated[str, Form(min_length=1)],
    person_id: Annotated[str, Form(min_length=1)],
    photos: Annotated[list[UploadFile], File(description="Three to five face photos")],
) -> PersonResponse:
    if not 3 <= len(photos) <= 5:
        raise HTTPException(status_code=422, detail="Registration requires 3 to 5 photos.")
    services = get_services()
    images = [await read_image_upload(photo) for photo in photos]
    if services.liveness is not None:
        liveness = services.liveness.check(images)
        if not liveness["live"]:
            raise HTTPException(
                status_code=422,
                detail=f"Registration blocked: spoof detected (score {liveness['score']:.2f}).",
            )
    try:
        person = services.registry.add(name, person_id, images)
    except (ValueError, KeyError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    return PersonResponse(id=person["id"], name=person["name"])


@app.post("/attendance", response_model=AnalyzeResponse)
async def attendance(
    frames: Annotated[list[UploadFile], File(description="Short liveness frame sequence")],
) -> AnalyzeResponse:
    if not 2 <= len(frames) <= 10:
        raise HTTPException(status_code=422, detail="Attendance requires 2 to 10 frames.")
    services = get_services()
    images = [await read_image_upload(frame) for frame in frames]
    if services.liveness is not None:
        liveness = services.liveness.check(images)
        if not liveness["live"]:
            raise HTTPException(
                status_code=422,
                detail=f"Attendance blocked: spoof detected (score {liveness['score']:.2f}).",
            )
    try:
        results = services.pipeline.analyze(images[-1])
    except (ValueError, RuntimeError) as error:
        raise HTTPException(status_code=422, detail=f"Could not analyze attendance: {error}") from error
    for result in results:
        if services.logs is not None:
            services.logs.log(
                person_id=result.get("person_id"),
                event_time=None,
                gender=result["gender"],
                age=result["age"],
                confidence=result["match_score"],
                source="attendance",
            )
    return AnalyzeResponse(faces=results)


@app.get("/people", response_model=list[PersonResponse])
def list_people() -> list[PersonResponse]:
    services = get_services()
    return [
        PersonResponse(id=person["id"], name=person["name"])
        for person in services.registry.list()
    ]


@app.delete("/people/{person_id}", status_code=204)
def delete_person(person_id: str) -> Response:
    services = get_services()
    try:
        services.registry.delete(person_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    return Response(status_code=204)


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", models_loaded=_services is not None)
