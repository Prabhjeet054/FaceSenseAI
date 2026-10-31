"""Person registration and face-embedding persistence."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from sqlalchemy import DateTime, LargeBinary, String, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.sql import func

from app.core.detector import FaceDetector
from app.core.embedder import EMBEDDING_DIMENSION, FaceEmbedder
from app.core.preprocess import preprocess_face
import config


MIN_PHOTOS = 3
MAX_PHOTOS = 5


class Base(DeclarativeBase):
    """Base for registry database models."""


class RegisteredPerson(Base):
    """Persisted registered identity and average face embedding."""

    __tablename__ = "registered_people"

    person_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    created_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[Any] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )


class FaceRegistry:
    """Register, update, delete, and list people by face embedding."""

    def __init__(
        self,
        database_url: str = f"sqlite:///{config.DATA_DIR / 'faces.db'}",
        detector: FaceDetector | Any | None = None,
        embedder: FaceEmbedder | Any | None = None,
    ) -> None:
        if database_url.startswith("sqlite:///"):
            Path(database_url.removeprefix("sqlite:///")).parent.mkdir(
                parents=True, exist_ok=True
            )
        self.engine = create_engine(database_url)
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)
        self.detector = detector or FaceDetector()
        self.embedder = embedder or FaceEmbedder()

    def add(
        self, name: str, person_id: str, photos: Sequence[str | Path | np.ndarray]
    ) -> dict[str, Any]:
        """Register a person from three to five photos."""
        self._validate_identity(name, person_id)
        embeddings = self._extract_embeddings(photos)
        with self.session_factory.begin() as session:
            if session.get(RegisteredPerson, person_id) is not None:
                raise ValueError(f"Person ID already registered: {person_id}")
            record = RegisteredPerson(
                person_id=person_id,
                name=name.strip(),
                embedding=self._serialize_embedding(self._average(embeddings)),
            )
            session.add(record)
        return self._record_to_dict(record)

    def update(
        self,
        person_id: str,
        name: str | None = None,
        photos: Sequence[str | Path | np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Update a registered name and optionally replace its photos."""
        if name is not None and not name.strip():
            raise ValueError("name must not be empty")
        if photos is not None:
            embeddings = self._extract_embeddings(photos)

        with self.session_factory.begin() as session:
            record = session.get(RegisteredPerson, person_id)
            if record is None:
                raise KeyError(f"Person ID is not registered: {person_id}")
            if name is not None:
                record.name = name.strip()
            if photos is not None:
                record.embedding = self._serialize_embedding(self._average(embeddings))
            session.add(record)
            session.flush()
            session.refresh(record)
        return self._record_to_dict(record)

    def delete(self, person_id: str) -> None:
        """Delete a registered person or raise if the ID does not exist."""
        with self.session_factory.begin() as session:
            record = session.get(RegisteredPerson, person_id)
            if record is None:
                raise KeyError(f"Person ID is not registered: {person_id}")
            session.delete(record)

    def list(self) -> list[dict[str, Any]]:
        """Return all registered people, including their stored embeddings."""
        with Session(self.engine) as session:
            records = session.scalars(
                select(RegisteredPerson).order_by(RegisteredPerson.person_id)
            ).all()
            return [self._record_to_dict(record) for record in records]

    def _extract_embeddings(
        self, photos: Sequence[str | Path | np.ndarray]
    ) -> list[np.ndarray]:
        if not MIN_PHOTOS <= len(photos) <= MAX_PHOTOS:
            raise ValueError(
                f"Registration requires between {MIN_PHOTOS} and {MAX_PHOTOS} photos"
            )

        embeddings = []
        for index, photo in enumerate(photos, start=1):
            image = self._load_photo(photo)
            detections = self.detector.detect(image)
            if not detections:
                raise ValueError(f"Photo {index} contains no face")
            if len(detections) > 1:
                raise ValueError(f"Photo {index} contains more than one face")
            detection = detections[0]
            aligned = preprocess_face(
                image,
                detection["bbox"],
                detection["landmarks"],
                model="recognition",
            )
            embedding = np.asarray(self.embedder.embed(aligned), dtype=np.float32)
            if embedding.shape != (EMBEDDING_DIMENSION,):
                raise ValueError(
                    f"Photo {index} produced an invalid embedding shape: {embedding.shape}"
                )
            embeddings.append(embedding)
        return embeddings

    @staticmethod
    def _load_photo(photo: str | Path | np.ndarray) -> np.ndarray:
        if isinstance(photo, np.ndarray):
            image = photo
        else:
            image = cv2.imread(str(photo))
        if image is None or image.size == 0:
            raise ValueError(f"Could not read photo: {photo}")
        return image

    @staticmethod
    def _average(embeddings: list[np.ndarray]) -> np.ndarray:
        average = np.mean(np.stack(embeddings), axis=0)
        norm = np.linalg.norm(average)
        if norm == 0 or not np.isfinite(norm):
            raise ValueError("Cannot store an invalid average embedding")
        return (average / norm).astype(np.float32)

    @staticmethod
    def _serialize_embedding(embedding: np.ndarray) -> bytes:
        return embedding.tobytes()

    @staticmethod
    def _record_to_dict(record: RegisteredPerson) -> dict[str, Any]:
        embedding = np.frombuffer(record.embedding, dtype=np.float32).copy()
        return {
            "id": record.person_id,
            "name": record.name,
            "embedding": embedding.tolist(),
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    @staticmethod
    def _validate_identity(name: str, person_id: str) -> None:
        if not name or not name.strip():
            raise ValueError("name must not be empty")
        if not person_id or not person_id.strip():
            raise ValueError("person_id must not be empty")


__all__ = ["FaceRegistry", "RegisteredPerson"]
