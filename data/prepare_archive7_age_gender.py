"""Prepare archive (7) and age_gender.csv for age and gender training."""

from __future__ import annotations

import argparse
from pathlib import Path
import csv
import tempfile
import zipfile

import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from data.prepare_dataset import IMAGE_SIZE
from data.prepare_age_datasets import extract_archive


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ARCHIVE = Path.home() / "Downloads" / "archive (7).zip"
DEFAULT_CSV = Path.home() / "Downloads" / "age_gender.csv"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_archive7"


def parse_gender(value: object) -> int:
    """Convert numeric or textual gender labels to the project convention."""
    normalized = str(value).strip().lower()
    if normalized in {"0", "male", "man", "men"}:
        return 0
    if normalized in {"1", "female", "woman", "women"}:
        return 1
    raise ValueError(f"Unsupported gender label: {value}")


def load_csv_records(csv_path: Path) -> list[dict[str, object]]:
    """Read exact age, gender, and pixel-vector records."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"age/gender CSV not found: {csv_path}")
    records: list[dict[str, object]] = []
    with csv_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"age", "gender", "pixels"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"{csv_path} must contain columns: {sorted(required)}")
        for row in reader:
            try:
                age = float(row["age"])
                gender = parse_gender(row["gender"])
                pixels = np.fromstring(row["pixels"], sep=" ", dtype=np.uint8)
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid record in {csv_path}: {row}") from error
            if not 0 <= age <= 100 or pixels.size != 48 * 48:
                continue
            records.append(
                {
                    "age": age,
                    "gender": gender,
                    "pixels": pixels.reshape(48, 48),
                    "source": "age_gender_csv",
                }
            )
    if not records:
        raise ValueError(f"No valid rows found in {csv_path}")
    return records


def load_archive_gender_records(root: Path) -> list[dict[str, object]]:
    """Read archive (7) gender labels and image paths."""
    labels_path = root / "gender_detection.csv"
    labels = pd.read_csv(labels_path)
    required = {"file", "gender", "split"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{labels_path} must contain columns: {sorted(required)}")
    records: list[dict[str, object]] = []
    for row in labels.itertuples(index=False):
        source = root / str(row.file)
        if not source.is_file():
            continue
        records.append(
            {
                "source": "archive7",
                "source_path": source,
                "gender": parse_gender(row.gender),
                "split": str(row.split),
            }
        )
    if not records:
        raise ValueError("No readable archive (7) gender images found")
    return records


def split_indices(count: int, random_state: int) -> dict[str, np.ndarray]:
    """Create reproducible train/validation/test indices."""
    indices = np.arange(count)
    train, remainder = train_test_split(
        indices, test_size=0.30, random_state=random_state
    )
    validation, test = train_test_split(
        remainder, test_size=0.50, random_state=random_state
    )
    return {"train": train, "validation": validation, "test": test}


def write_image(image: np.ndarray, destination: Path) -> None:
    resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(destination), resized):
        raise OSError(f"Could not write image: {destination}")


def prepare(
    archive_path: Path = DEFAULT_ARCHIVE,
    csv_path: Path = DEFAULT_CSV,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    random_state: int = 42,
) -> dict[str, int]:
    """Create separate age and gender datasets with compatible labels.csv files."""
    with tempfile.TemporaryDirectory(prefix="facesense-archive7-") as temporary:
        archive_root = extract_archive(archive_path, Path(temporary) / "archive7")
        csv_records = load_csv_records(csv_path)
        archive_records = load_archive_gender_records(archive_root)

        age_root = output_root / "age"
        gender_root = output_root / "gender"
        for root in (age_root, gender_root):
            root.mkdir(parents=True, exist_ok=True)

        csv_splits = split_indices(len(csv_records), random_state)
        age_rows: list[dict[str, object]] = []
        gender_rows: list[dict[str, object]] = []
        for split, indices in csv_splits.items():
            split_dir_age = age_root / split
            split_dir_gender = gender_root / split
            split_dir_age.mkdir(parents=True, exist_ok=True)
            split_dir_gender.mkdir(parents=True, exist_ok=True)
            for index in indices:
                record = csv_records[int(index)]
                filename = f"csv_{int(index):06d}.jpg"
                image = record["pixels"]
                write_image(image, split_dir_age / filename)
                write_image(image, split_dir_gender / filename)
                age_rows.append(
                    {
                        "filename": filename,
                        "age": record["age"],
                        "gender": record["gender"],
                        "split": split,
                    }
                )
                gender_rows.append(
                    {
                        "filename": filename,
                        "gender": record["gender"],
                        "split": split,
                    }
                )

        archive_splits = {
            str(split): [row for row in archive_records if row["split"] == split]
            for split in {"train", "validation", "test"}
        }
        for split, records in archive_splits.items():
            split_dir = gender_root / split
            for index, record in enumerate(records):
                filename = f"archive7_{split}_{index:04d}.jpg"
                image = cv2.imread(str(record["source_path"]))
                if image is None:
                    continue
                write_image(image, split_dir / filename)
                gender_rows.append(
                    {
                        "filename": filename,
                        "gender": record["gender"],
                        "split": split,
                    }
                )

        pd.DataFrame(age_rows).to_csv(age_root / "labels.csv", index=False)
        pd.DataFrame(gender_rows).to_csv(gender_root / "labels.csv", index=False)
        summary = {
            "age_images": len(age_rows),
            "gender_images": len(gender_rows),
            "archive7_gender_images": sum(len(rows) for rows in archive_splits.values()),
        }
        print(summary)
        return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-path", type=Path, default=DEFAULT_ARCHIVE)
    parser.add_argument("--csv-path", type=Path, default=DEFAULT_CSV)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(**vars(parse_args()))
