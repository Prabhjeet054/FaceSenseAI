from datetime import datetime, timedelta, timezone

from app.services.logs import RecognitionLogService


def test_recognition_log_service_filters_and_groups_events() -> None:
    service = RecognitionLogService(database_url="sqlite:///:memory:")
    service.log(
        person_id="person-1",
        event_time=datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc),
        gender="male",
        age=27,
        confidence=0.94,
        source="webcam",
    )
    service.log(
        person_id="person-2",
        event_time=datetime(2026, 10, 31, 9, 0, tzinfo=timezone.utc),
        gender="female",
        age=41,
        confidence=0.88,
        source="api",
    )

    events = service.list(date_from=datetime(2026, 10, 31).date(), person_id="person-2")
    assert len(events) == 1
    assert events[0]["person_id"] == "person-2"

    stats = service.stats(
        date_from=datetime(2026, 10, 30).date(),
        date_to=datetime(2026, 10, 31).date(),
    )
    assert stats == {
        "total": 2,
        "by_gender": {"male": 1, "female": 1},
        "by_age_group": {"young adult": 1, "adult": 1},
        "by_day": {"2026-10-30": 1, "2026-10-31": 1},
    }


def test_twenty_events_filters_and_stats_totals() -> None:
    service = RecognitionLogService(database_url="sqlite:///:memory:")
    start = datetime(2026, 10, 28, 9, 0, tzinfo=timezone.utc)

    for index in range(20):
        service.log(
            person_id="person-1" if index < 12 else "person-2",
            event_time=start + timedelta(days=index // 5, minutes=index),
            gender="male" if index % 2 == 0 else "female",
            age=25 if index % 2 == 0 else 40,
            confidence=0.8,
            source="test",
        )

    assert len(service.list()) == 20
    assert len(service.list(person_id="person-1")) == 12
    assert len(service.list(person_id="person-2")) == 8
    assert len(
        service.list(
            date_from=datetime(2026, 10, 29).date(),
            date_to=datetime(2026, 10, 30).date(),
        )
    ) == 10
    assert len(
        service.list(
            date_from=datetime(2026, 10, 29).date(),
            date_to=datetime(2026, 10, 30).date(),
            person_id="person-1",
        )
    ) == 7

    stats = service.stats()
    assert stats["total"] == 20
    assert stats["by_gender"] == {"male": 10, "female": 10}
    assert stats["by_age_group"] == {"young adult": 10, "adult": 10}
    assert stats["by_day"] == {
        "2026-10-28": 5,
        "2026-10-29": 5,
        "2026-10-30": 5,
        "2026-10-31": 5,
    }

    filtered_stats = service.stats(
        date_from=datetime(2026, 10, 29).date(),
        date_to=datetime(2026, 10, 30).date(),
        person_id="person-1",
    )
    assert filtered_stats["total"] == 7
    assert sum(filtered_stats["by_gender"].values()) == 7
    assert sum(filtered_stats["by_age_group"].values()) == 7
    assert sum(filtered_stats["by_day"].values()) == 7
