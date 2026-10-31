from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import ui.app as dashboard


APP_PATH = Path(__file__).parents[2] / "ui" / "app.py"


def run_page(page: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH))
    app.run()
    app.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize(
    "page",
    [
        "Dashboard",
        "Daily Attendance",
        "Upload and Analyze",
        "Live Camera",
        "Register Person",
        "Logs and Stats",
    ],
)
def test_each_dashboard_page_loads_without_errors(page: str) -> None:
    app = run_page(page)

    assert not app.exception
    assert not app.error
    assert app.header[0].value == page


def test_upload_flow_renders_results_table_with_mocked_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    image = np.full((32, 32, 3), 127, dtype=np.uint8)
    success, encoded = cv2.imencode(".jpg", image)
    assert success

    class FakeUpload:
        name = "face.jpg"

        def getvalue(self) -> bytes:
            return encoded.tobytes()

    monkeypatch.setattr(
        dashboard,
        "backend_request",
        lambda method, path, **kwargs: {
            "faces": [
                {
                    "box": [2, 2, 20, 20],
                    "name": "Rahul",
                    "person_id": "person-1",
                    "match_score": 0.94,
                    "gender": "male",
                    "gender_confidence": 0.94,
                    "age": 27,
                    "age_group": "young adult",
                }
            ]
        },
    )
    monkeypatch.setattr(
        dashboard.st,
        "file_uploader",
        lambda *args, **kwargs: FakeUpload(),
    )

    def upload_page_with_mock() -> None:
        import ui.app as dashboard

        dashboard.upload_and_analyze_page()

    app = AppTest.from_function(upload_page_with_mock).run()
    assert not app.exception
    assert len(app.button) == 1

    app.button[0].click().run()

    assert not app.exception
    assert len(app.dataframe) == 1
    assert "Rahul" in str(app.dataframe[0].value)


def test_dashboard_renders_charts_table_and_csv_export(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = [
        {
            "id": 1,
            "person_id": "person-1",
            "time": "2026-10-31T09:15:00+00:00",
            "gender": "male",
            "age": 27,
            "age_group": "young adult",
            "confidence": 0.94,
            "source": "api",
        }
    ]

    def fake_backend_request(method: str, path: str, **kwargs: object) -> object:
        if path == "/logs":
            return logs
        if path == "/stats":
            return {
                "total": 1,
                "by_gender": {"male": 1},
                "by_age_group": {"young adult": 1},
                "by_day": {"2026-10-31": 1},
            }
        raise AssertionError(f"Unexpected endpoint: {path}")

    monkeypatch.setattr(dashboard, "backend_request", fake_backend_request)

    def dashboard_with_mock() -> None:
        import ui.app as dashboard

        dashboard.dashboard_page()

    app = AppTest.from_function(dashboard_with_mock).run()
    app.button[0].click().run()

    assert not app.exception
    assert len(app.dataframe) == 1
    assert "person-1" in str(app.dataframe[0].value)


def test_chart_builders_return_expected_data_points() -> None:
    logs = pd.DataFrame(
        [
            {"gender": "male", "age_group": "young adult", "time": "2026-10-31T09:00:00Z"},
            {"gender": "female", "age_group": "adult", "time": "2026-10-31T09:00:00Z"},
            {"gender": "female", "age_group": "adult", "time": "2026-10-31T18:00:00Z"},
        ]
    )

    gender_figure = dashboard.build_gender_chart(logs).to_dict()
    age_figure = dashboard.build_age_group_chart(logs).to_dict()
    hourly_figure = dashboard.build_hourly_chart(logs).to_dict()

    def data_point_count(figure: dict[str, object]) -> int:
        datasets = figure["datasets"]
        assert isinstance(datasets, dict)
        values = next(iter(datasets.values()))
        assert isinstance(values, list)
        return len(values)

    assert data_point_count(gender_figure) == 2
    assert data_point_count(age_figure) == 2
    assert data_point_count(hourly_figure) == 24
