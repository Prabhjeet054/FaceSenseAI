"""Prepare exact-age and age-band archives for age-regression training."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import tempfile
import zipfile

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

from data.prepare_dataset import IMAGE_SIZE


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE_5 = Path.home() / "Downloads" / "archive (5).zip"
DEFAULT_ARCHIVE_4 = Path.home() / "Downloads" / "archive (4).zip"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
AGE_BAND = re.compile(r"^(?P<low>\d+)-(?P<high>\d+)$")


def extract_archive(archive: Path, destination: Path) -> Path:
    """Extract an archive after rejecting path traversal entries."""
    if not archive.is_file():
        raise FileNotFoundError(f"Dataset archive not found: {archive}")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as source:
        root = destination.resolve()
        for member in source.infolist():
            target = (destination / member.filename).resolve()
            if target != root and root not in target.parents:
                raise ValueError(f"Archive contains an unsafe path: {member.filename}")
        source.extractall(destination)
    return destination


def parse_exact_age(path: Path) -> tuple[float, int] | None:
    """Parse age and gender from a UTKFace-style filename."""
    parts = path.stem.split("_", 3)
    if len(parts) != 4:
        return None
    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None
    if not 0 <= age <= 100 or gender not in (0, 1):
        return None
    return float(age), gender


def parse_age_band(value: object) -> tuple[float, float] | None:
    """Return the bounds of a grouped age label such as ``21-30``."""
    match = AGE_BAND.fullmatch(str(value).strip())
    if match is None:
        return None
    low, high = int(match["low"]), int(match["high"])
    if low < 0 or high < low or high > 100:
        return None
    return float(low), float(high)


def collect_records(archive_5_dir: Path, archive_4_dir: Path) -> pd.DataFrame:
    """Collect exact UTKFace labels and lower-confidence grouped labels."""
    records: list[dict[str, object]] = []
    for path in sorted(archive_5_dir.rglob("*")):
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parsed = parse_exact_age(path)
        if parsed is None:
            continue
        age, gender = parsed
        records.append(
            {
                "source_path": str(path),
                "age": age,
                "gender": gender,
                "label_weight": 1.0,
                "label_source": "exact",
            }
        )

    labels_path = archive_4_dir / "age_detection.csv"
    if not labels_path.is_file():
        raise FileNotFoundError(f"Grouped age labels not found: {labels_path}")
    grouped = pd.read_csv(labels_path)
    required = {"file", "age"}
    if not required.issubset(grouped.columns):
        raise ValueError(f"{labels_path} must contain columns: {sorted(required)}")
    for row in grouped.itertuples(index=False):
        bounds = parse_age_band(row.age)
        source_path = archive_4_dir / str(row.file)
        if bounds is None or not source_path.is_file():
            continue
        low, high = bounds
        records.append(
            {
                "source_path": str(source_path),
                "age": (low + high) / 2.0,
                "gender": -1,
                "label_weight": 0.35,
                "label_source": "age_band_midpoint",
            }
        )

    if not records:
        raise ValueError("No valid age-labelled images were found in the archives")
    return pd.DataFrame(records)


def split_records(records: pd.DataFrame, random_state: int) -> dict[str, pd.DataFrame]:
    """Split records while preserving broad age distribution."""
    bins = pd.cut(
        records["age"],
        bins=[-1, 12, 19, 35, 59, 100],
        labels=False,
        include_lowest=True,
    )
    try:
        train, remainder = train_test_split(
            records, test_size=0.30, random_state=random_state, stratify=bins
        )
        remainder_bins = bins.loc[remainder.index]
        validation, test = train_test_split(
            remainder,
            test_size=0.50,
            random_state=random_state,
            stratify=remainder_bins,
        )
    except ValueError:
        train, remainder = train_test_split(
            records, test_size=0.30, random_state=random_state
        )
        validation, test = train_test_split(
            remainder, test_size=0.50, random_state=random_state
        )
    return {"train": train, "validation": validation, "test": test}


def prepare_age_dataset(
    archive_5: Path = DEFAULT_ARCHIVE_5,
    archive_4: Path = DEFAULT_ARCHIVE_4,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    random_state: int = 42,
) -> pd.DataFrame:
    """Extract, combine, resize, and split both requested datasets."""
    with tempfile.TemporaryDirectory(prefix="facesense-age-data-") as temporary:
        root = Path(temporary)
        exact_dir = extract_archive(archive_5, root / "archive_5")
        grouped_dir = extract_archive(archive_4, root / "archive_4")
        records = collect_records(exact_dir, grouped_dir)
        splits = split_records(records, random_state)

        output_dir.mkdir(parents=True, exist_ok=True)
        rows: list[dict[str, object]] = []
        for split, split_records_frame in splits.items():
            split_dir = output_dir / split
            split_dir.mkdir(parents=True, exist_ok=True)
            for index, row in split_records_frame.reset_index(drop=True).iterrows():
                image = cv2.imread(str(row["source_path"]))
                if image is None:
                    continue
                filename = (
                    f"{row['label_source']}_{split}_{index:06d}"
                    f"{Path(str(row['source_path'])).suffix.lower()}"
                )
                destination = split_dir / filename
                resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
                if not cv2.imwrite(str(destination), resized):
                    raise OSError(f"Could not write processed image: {destination}")
                rows.append(
                    {
                        "filename": filename,
                        "age": float(row["age"]),
                        "gender": int(row["gender"]),
                        "label_weight": float(row["label_weight"]),
                        "label_source": row["label_source"],
                        "split": split,
                    }
                )

        if not rows:
            raise ValueError("No readable images were written to the processed dataset")
        labels = pd.DataFrame(rows).sort_values("filename")
        labels.to_csv(output_dir / "labels.csv", index=False)
        print(f"Prepared {len(labels)} images in {output_dir}")
        print(labels.groupby(["label_source", "split"]).size().to_string())
        return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-5", type=Path, default=DEFAULT_ARCHIVE_5)
    parser.add_argument("--archive-4", type=Path, default=DEFAULT_ARCHIVE_4)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare_age_dataset(**vars(parse_args()))
