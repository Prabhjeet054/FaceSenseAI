"""Persistence and reporting for face-recognition events."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from typing import Any

from sqlalchemy import DateTime, Float, String, create_engine, func, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from app.models.age_model import predict_age_group
from app.services.registry import Base


class RecognitionEvent(Base):
    """A single face-recognition event."""

    __tablename__ = "recognition_events"

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=True)
    person_id: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    event_time: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), nullable=False, index=True
    )
    gender: Mapped[str] = mapped_column(String(32), nullable=False)
    age: Mapped[float] = mapped_column(Float, nullable=False)
    age_group: Mapped[str] = mapped_column(String(32), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)


class RecognitionLogService:
    """Store recognition events and produce filtered summaries."""

    def __init__(
        self,
        database_url: str | None = None,
        engine: Engine | None = None,
    ) -> None:
        if engine is None:
            if database_url is None:
                raise ValueError("database_url or engine is required")
            engine_options: dict[str, Any] = {}
            if database_url == "sqlite:///:memory:":
                engine_options = {
                    "connect_args": {"check_same_thread": False},
                    "poolclass": StaticPool,
                }
            engine = create_engine(database_url, **engine_options)
        self.engine = engine
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def log(
        self,
        person_id: str | None,
        event_time: datetime | None,
        gender: str,
        age: float,
        confidence: float,
        source: str,
    ) -> dict[str, Any]:
        """Persist one recognition event and return its public representation."""
        if not 0 <= confidence <= 1:
            raise ValueError("confidence must be between 0 and 1")
        if not source.strip():
            raise ValueError("source must not be empty")
        timestamp = event_time or datetime.now(timezone.utc)
        event = RecognitionEvent(
            person_id=person_id,
            event_time=timestamp,
            gender=gender,
            age=float(age),
            age_group=predict_age_group(age),
            confidence=float(confidence),
            source=source.strip(),
        )
        with self.session_factory.begin() as session:
            session.add(event)
            session.flush()
        return self._to_dict(event)

    def list(
        self,
        date_from: date | None = None,
        date_to: date | None = None,
        person_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return events filtered by inclusive dates and person ID."""
        start, end = self._date_bounds(date_from, date_to)
        statement = select(RecognitionEvent).order_by(RecognitionEvent.event_time.desc())
        if start is not None:
            statement = statement.where(RecognitionEvent.event_time >= start)
        if end is not None:
            statement = statement.where(RecognitionEvent.event_time < end)
        if person_id is not None:
            statement = statement.where(RecognitionEvent.person_id == person_id)
        with Session(self.engine) as session:
            return [self._to_dict(event) for event in session.scalars(statement)]

    def stats(
        self,
        date_from: date | None = None,
        date_to: date | None = None,
        person_id: str | None = None,
    ) -> dict[str, Any]:
        """Return event totals grouped by gender, age group, and calendar day."""
        events = self.list(date_from=date_from, date_to=date_to, person_id=person_id)
        by_gender: dict[str, int] = {}
        by_age_group: dict[str, int] = {}
        by_day: dict[str, int] = {}
        for event in events:
            by_gender[event["gender"]] = by_gender.get(event["gender"], 0) + 1
            by_age_group[event["age_group"]] = (
                by_age_group.get(event["age_group"], 0) + 1
            )
            day = event["time"][:10]
            by_day[day] = by_day.get(day, 0) + 1
        return {
            "total": len(events),
            "by_gender": by_gender,
            "by_age_group": by_age_group,
            "by_day": dict(sorted(by_day.items())),
        }

    @staticmethod
    def _date_bounds(
        date_from: date | None, date_to: date | None
    ) -> tuple[datetime | None, datetime | None]:
        if date_from is not None and date_to is not None and date_from > date_to:
            raise ValueError("date_from must not be after date_to")
        start = (
            datetime.combine(date_from, time.min, tzinfo=timezone.utc)
            if date_from
            else None
        )
        end = (
            datetime.combine(date_to + timedelta(days=1), time.min, tzinfo=timezone.utc)
            if date_to
            else None
        )
        return start, end

    @staticmethod
    def _to_dict(event: RecognitionEvent) -> dict[str, Any]:
        timestamp = event.event_time
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return {
            "id": event.id,
            "person_id": event.person_id,
            "time": timestamp.isoformat(),
            "gender": event.gender,
            "age": event.age,
            "age_group": event.age_group,
            "confidence": event.confidence,
            "source": event.source,
        }


__all__ = ["RecognitionEvent", "RecognitionLogService"]
