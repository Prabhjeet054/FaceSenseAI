"""Prepare UTKFace images for training age and gender models."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import cv2
import pandas as pd
from sklearn.model_selection import train_test_split


IMAGE_SIZE = (128, 128)
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT_DIR = PROJECT_ROOT / "data" / "raw"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


def parse_filename(path: Path) -> tuple[int, int] | None:
    """Return age and gender from an UTKFace filename, if it is valid."""
    parts = path.stem.split("_", 3)
    if len(parts) != 4:
        return None

    try:
        age, gender = int(parts[0]), int(parts[1])
    except ValueError:
        return None

    if age < 0 or gender not in (0, 1):
        return None
    return age, gender


def find_images(input_dir: Path) -> Iterable[Path]:
    """Yield supported image files directly under the input directory."""
    extensions = {".jpg", ".jpeg", ".png"}
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def split_records(
    records: pd.DataFrame, random_state: int
) -> dict[str, pd.DataFrame]:
    """Split records into 70/15/15 partitions stratified by gender."""
    train, remainder = train_test_split(
        records,
        test_size=0.30,
        random_state=random_state,
        stratify=records["gender"],
    )
    validation, test = train_test_split(
        remainder,
        test_size=0.50,
        random_state=random_state,
        stratify=remainder["gender"],
    )
    return {"train": train, "validation": validation, "test": test}


def prepare_dataset(
    input_dir: Path,
    output_dir: Path,
    random_state: int,
) -> pd.DataFrame:
    """Read, resize, split, and label valid UTKFace images."""
    records: list[dict[str, int | str]] = []
    skipped = 0

    for image_path in find_images(input_dir):
        parsed = parse_filename(image_path)
        if parsed is None:
            skipped += 1
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        resized = cv2.resize(image, IMAGE_SIZE, interpolation=cv2.INTER_AREA)
        records.append(
            {
                "filename": image_path.name,
                "age": parsed[0],
                "gender": parsed[1],
                "_source": str(image_path),
                "_image": resized,
            }
        )

    if not records:
        raise ValueError(f"No valid UTKFace images found in {input_dir}")

    metadata = pd.DataFrame(records)
    try:
        splits = split_records(metadata, random_state)
    except ValueError as error:
        raise ValueError(
            "Unable to create stratified 70/15/15 splits. "
            "Each gender must have enough valid images."
        ) from error

    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[pd.DataFrame] = []
    for split_name, split_records_frame in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        split_records_frame = split_records_frame.copy()
        split_records_frame["split"] = split_name

        for filename, image in zip(
            split_records_frame["filename"], split_records_frame["_image"]
        ):
            destination = split_dir / filename
            if not cv2.imwrite(str(destination), image):
                raise OSError(f"Could not write processed image: {destination}")

        rows.append(split_records_frame[["filename", "age", "gender", "split"]])

    labels = pd.concat(rows, ignore_index=True).sort_values("filename")
    labels.to_csv(output_dir / "labels.csv", index=False)
    print(f"Processed {len(labels)} images; skipped {skipped}")
    print(labels["split"].value_counts().sort_index().to_string())
    return labels


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing UTKFace images (default: data/raw)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed images and labels.csv (default: data/processed)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Seed used for reproducible stratified splits (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.input_dir, args.output_dir, args.random_state)
