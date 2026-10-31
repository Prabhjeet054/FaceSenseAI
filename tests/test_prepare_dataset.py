from pathlib import Path

import pytest

from data.prepare_dataset import parse_filename


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
