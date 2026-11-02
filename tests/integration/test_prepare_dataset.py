from pathlib import Path

import pytest

from data.prepare_dataset import parse_filename
from data.prepare_age_datasets import parse_age_band, parse_exact_age


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("25_0_2_20170116174525125.jpg", (25, 0)),
        ("3_1_0_20170109150557335.jpg", (3, 1)),
        ("80_0_3_20170111201827993.jpg", (80, 0)),
        ("42_1_1_20170117005207889.jpeg", (42, 1)),
        ("not-a-utkface-name.jpg", None),
    ],
)
def test_parse_filename_extracts_age_and_gender_or_handles_malformed_name(
    filename: str, expected: tuple[int, int] | None
) -> None:
    assert parse_filename(Path(filename)) == expected


def test_parse_filename_does_not_raise_for_malformed_name() -> None:
    assert parse_filename(Path("malformed.jpg")) is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("25_0_2_20170116174525125.jpg", (25.0, 0)),
        ("90_1_0_20170120230312163.jpg", (90.0, 1)),
        ("not-a-utkface-name.jpg", None),
    ],
)
def test_parse_exact_age_supports_nested_utkface_files(
    filename: str, expected: tuple[float, int] | None
) -> None:
    assert parse_exact_age(Path(filename)) == expected


@pytest.mark.parametrize(
    ("label", "expected"),
    [("18-20", (18.0, 20.0)), ("51-60", (51.0, 60.0)), ("unknown", None)],
)
def test_parse_age_band_returns_interval_bounds(
    label: str, expected: tuple[float, float] | None
) -> None:
    assert parse_age_band(label) == expected
