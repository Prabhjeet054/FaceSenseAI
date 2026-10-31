from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

streamlit = pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest

import ui.app as dashboard


APP_PATH = Path(__file__).parents[1] / "ui" / "app.py"


def run_page(page: str) -> AppTest:
    app = AppTest.from_file(str(APP_PATH))
    app.run()
    app.radio[0].set_value(page).run()
    return app


@pytest.mark.parametrize(
    "page",
    ["Upload and Analyze", "Live Camera", "Register Person", "Logs and Stats"],
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
