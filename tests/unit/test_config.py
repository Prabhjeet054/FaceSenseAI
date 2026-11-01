from numbers import Real
from pathlib import Path

import config


def test_path_constants_exist_or_can_be_created() -> None:
    path_constants = {
        name: value
        for name, value in vars(config).items()
        if name.isupper() and isinstance(value, Path)
    }

    assert path_constants

    for name, path in path_constants.items():
        if path.exists():
            assert path.is_file() or path.is_dir(), f"{name} is not a valid path: {path}"
        else:
            path.mkdir(parents=True)
            assert path.is_dir(), f"{name} could not be created: {path}"


def test_threshold_constants_are_numbers_between_zero_and_one() -> None:
    threshold_constants = {
        name: value
        for name, value in vars(config).items()
        if name.isupper()
        and ("THRESHOLD" in name or "CONFIDENCE" in name)
    }

    assert threshold_constants

    for name, threshold in threshold_constants.items():
        assert isinstance(threshold, Real) and not isinstance(threshold, bool), (
            f"{name} must be a number"
        )
        assert 0 <= threshold <= 1, f"{name} must be between 0 and 1"
