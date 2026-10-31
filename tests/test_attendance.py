from datetime import datetime, timedelta, timezone

import pandas as pd

from app.services.attendance import AttendanceService


def test_attendance_marks_once_per_day_and_applies_cooldown() -> None:
    service = AttendanceService(
        database_url="sqlite:///:memory:",
        cooldown_seconds=60,
    )
    first_time = datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc)

    first = service.mark_present("person-1", first_time)
    repeat = service.mark_present("person-1", first_time + timedelta(seconds=30))
    later_same_day = service.mark_present(
        "person-1", first_time + timedelta(hours=2)
    )
    next_day = service.mark_present(
        "person-1", first_time + timedelta(days=1)
    )

    assert first["marked"] is True
    assert first["reason"] == "first_recognition"
    assert repeat["marked"] is False
    assert repeat["reason"] == "cooldown"
    assert later_same_day["marked"] is False
    assert later_same_day["reason"] == "already_present"
    assert next_day["marked"] is True
    assert len(service.list_for_day(first_time.date())) == 1
    assert len(service.list_for_day((first_time + timedelta(days=1)).date())) == 1


def test_attendance_excel_export_contains_daily_sheet() -> None:
    service = AttendanceService(database_url="sqlite:///:memory:")
    timestamp = datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc)
    service.mark_present("person-1", timestamp)

    workbook = service.export_excel(timestamp.date())
    frame = pd.read_excel(pd.io.common.BytesIO(workbook), sheet_name="Attendance")

    assert list(frame.columns) == ["person_id", "date", "time", "source"]
    assert frame["person_id"].tolist() == ["person-1"]


def test_five_same_day_recognitions_create_one_row_and_export_columns() -> None:
    service = AttendanceService(database_url="sqlite:///:memory:")
    first_time = datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc)

    results = [
        service.mark_present(
            "person-1",
            first_time + timedelta(minutes=recognition * 30),
        )
        for recognition in range(5)
    ]

    assert results[0]["marked"] is True
    assert all(result["marked"] is False for result in results[1:])
    rows = service.list_for_day(first_time.date())
    assert len(rows) == 1
    assert rows[0]["person_id"] == "person-1"

    workbook = service.export_excel(first_time.date())
    frame = pd.read_excel(pd.io.common.BytesIO(workbook), sheet_name="Attendance")
    assert list(frame.columns) == ["person_id", "date", "time", "source"]
    assert len(frame) == 1
