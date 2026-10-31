"""Daily attendance persistence and Excel export."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta, timezone
from io import BytesIO
from typing import Any

import pandas as pd
from sqlalchemy import DateTime, Integer, String, UniqueConstraint, create_engine, delete, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool

from app.services.registry import Base
import config


class AttendanceRecord(Base):
    """The first successful attendance recognition for a person on a day."""

    __tablename__ = "attendance_records"
    __table_args__ = (
        UniqueConstraint("person_id", "attendance_date", name="uq_attendance_person_day"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    person_id: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    attendance_date: Mapped[date] = mapped_column(nullable=False, index=True)
    marked_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source: Mapped[str] = mapped_column(String(64), nullable=False)


class AttendanceService:
    """Mark each registered person present once per day."""

    def __init__(
        self,
        database_url: str | None = None,
        engine: Engine | None = None,
        cooldown_seconds: int = config.ATTENDANCE_DUPLICATE_WINDOW_SECONDS,
    ) -> None:
        if cooldown_seconds < 0:
            raise ValueError("cooldown_seconds must not be negative")
        if engine is None:
            if database_url is None:
                raise ValueError("database_url or engine is required")
            options: dict[str, Any] = {}
            if database_url == "sqlite:///:memory:":
                options = {
                    "connect_args": {"check_same_thread": False},
                    "poolclass": StaticPool,
                }
            engine = create_engine(database_url, **options)
        self.engine = engine
        self.cooldown_seconds = cooldown_seconds
        Base.metadata.create_all(self.engine)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def mark_present(
        self,
        person_id: str,
        marked_at: datetime | None = None,
        source: str = "attendance",
    ) -> dict[str, Any]:
        """Mark first attendance, or report why a repeat was ignored."""
        if not person_id.strip():
            raise ValueError("person_id must not be empty")
        timestamp = marked_at or datetime.now(timezone.utc)
        day = timestamp.date()
        with self.session_factory.begin() as session:
            statement = select(AttendanceRecord).where(
                AttendanceRecord.person_id == person_id,
                AttendanceRecord.attendance_date == day,
            )
            existing = session.scalar(statement)
            if existing is not None:
                existing_time = existing.marked_at
                if existing_time.tzinfo is None:
                    existing_time = existing_time.replace(tzinfo=timezone.utc)
                elapsed = (timestamp - existing_time).total_seconds()
                reason = (
                    "cooldown"
                    if elapsed < self.cooldown_seconds
                    else "already_present"
                )
                return {
                    "marked": False,
                    "reason": reason,
                    **self._to_dict(existing),
                }
            record = AttendanceRecord(
                person_id=person_id,
                attendance_date=day,
                marked_at=timestamp,
                source=source.strip() or "attendance",
            )
            session.add(record)
            session.flush()
            return {"marked": True, "reason": "first_recognition", **self._to_dict(record)}

    def list_for_day(self, attendance_date: date | None = None) -> list[dict[str, Any]]:
        day = attendance_date or datetime.now(timezone.utc).date()
        statement = select(AttendanceRecord).where(
            AttendanceRecord.attendance_date == day
        ).order_by(AttendanceRecord.marked_at)
        with Session(self.engine) as session:
            return [self._to_dict(record) for record in session.scalars(statement)]

    def delete_for_person(self, person_id: str) -> int:
        with self.session_factory.begin() as session:
            result = session.execute(
                delete(AttendanceRecord).where(AttendanceRecord.person_id == person_id)
            )
            return int(result.rowcount or 0)

    def export_excel(self, attendance_date: date | None = None) -> bytes:
        rows = self.list_for_day(attendance_date)
        frame = pd.DataFrame(
            rows,
            columns=["person_id", "date", "time", "source"],
        )
        output = BytesIO()
        with pd.ExcelWriter(output, engine="openpyxl") as writer:
            frame.to_excel(writer, index=False, sheet_name="Attendance")
        return output.getvalue()

    @staticmethod
    def _to_dict(record: AttendanceRecord) -> dict[str, Any]:
        marked_at = record.marked_at
        if marked_at.tzinfo is None:
            marked_at = marked_at.replace(tzinfo=timezone.utc)
        return {
            "id": record.id,
            "person_id": record.person_id,
            "date": record.attendance_date.isoformat(),
            "time": marked_at.isoformat(),
            "source": record.source,
        }


__all__ = ["AttendanceRecord", "AttendanceService"]
