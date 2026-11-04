"""Prepare FairFace and Spring 2021 assignment data for FaceSense training."""

from __future__ import annotations

import argparse
import tempfile
import zipfile
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_FAIRFACE_ROOT = Path.home() / "Downloads" / "FairFace"
DEFAULT_ASSIGNMENT_ARCHIVE = (
    Path.home() / "Downloads" / "spring-2021-dl-assignment-2.zip"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "processed_fairface"

AGE_MIDPOINTS = {
    "0-2": 1.0,
    "3-9": 6.0,
    "10-19": 14.5,
    "20-29": 24.5,
    "30-39": 34.5,
    "40-49": 44.5,
    "50-59": 54.5,
    "60-69": 64.5,
    "more than 70": 75.0,
}


def parse_gender(value: object) -> int:
    normalized = str(value).strip().lower()
    if normalized in {"male", "man", "0", "0.0"}:
        return 0
    if normalized in {"female", "woman", "1", "1.0"}:
        return 1
    raise ValueError(f"Unsupported gender label: {value}")


def load_fairface(path: Path, split: str) -> list[dict[str, object]]:
    labels = pd.read_csv(path)
    required = {"file", "age", "gender"}
    if not required.issubset(labels.columns):
        raise ValueError(f"{path} must contain columns: {sorted(required)}")
    rows = []
    for row in labels.itertuples(index=False):
        age_label = str(row.age).strip().lower()
        if age_label not in AGE_MIDPOINTS:
            continue
        source = path.parent / str(row.file)
        if source.is_file():
            rows.append(
                {
                    "source_path": str(source.resolve()),
                    "age": AGE_MIDPOINTS[age_label],
                    "gender": parse_gender(row.gender),
                    "split": split,
                    "label_weight": 0.65,
                    "source": "fairface",
                }
            )
    if not rows:
        raise ValueError(f"No usable FairFace rows found in {path}")
    return rows


def extract_assignment_gender(
    archive_path: Path, temporary_root: Path
) -> list[dict[str, object]]:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(temporary_root)
        labels = pd.read_csv(temporary_root / "wiki_judge.csv")
        rows = []
        image_root = temporary_root / "wiki_judge_images" / "wiki_judge_images"
        for row in labels.itertuples(index=False):
            if pd.isna(row.gender):
                continue
            image = image_root / f"{int(row.ID)}.png"
            if image.is_file():
                rows.append(
                    {
                        "source_path": str(image.resolve()),
                        "gender": parse_gender(row.gender),
                        "source": "spring_2021_dl_assignment_2",
                    }
                )
        return rows


def prepare(
    fairface_root: Path = DEFAULT_FAIRFACE_ROOT,
    assignment_archive: Path = DEFAULT_ASSIGNMENT_ARCHIVE,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, int]:
    output_root.mkdir(parents=True, exist_ok=True)
    fairface_rows = load_fairface(
        fairface_root / "fairface_label_train.csv", "train"
    ) + load_fairface(fairface_root / "fairface_label_val.csv", "validation")

    with tempfile.TemporaryDirectory(prefix="facesense-assignment-") as temporary:
        assignment_rows = extract_assignment_gender(
            assignment_archive, Path(temporary)
        )
        train_rows, validation_rows = train_test_split(
            assignment_rows, test_size=0.25, random_state=42, stratify=[
                row["gender"] for row in assignment_rows
            ],
        )
        for row in train_rows:
            row["split"] = "train"
        for row in validation_rows:
            row["split"] = "validation"

        # Keep extracted assignment images available after the temporary archive
        # directory is removed.
        assignment_dir = output_root / "assignment_images"
        assignment_dir.mkdir(exist_ok=True)
        all_rows = fairface_rows + train_rows + validation_rows
        for index, row in enumerate(train_rows + validation_rows):
            destination = assignment_dir / f"assignment_{index:04d}.png"
            destination.write_bytes(Path(row["source_path"]).read_bytes())
            row["source_path"] = str(destination)

        age_rows = [
            row for row in fairface_rows if row["split"] in {"train", "validation"}
        ]
        gender_rows = [
            {
                "source_path": row["source_path"],
                "gender": row["gender"],
                "split": row["split"],
                "source": row["source"],
            }
            for row in all_rows
        ]
        (output_root / "age").mkdir(exist_ok=True)
        (output_root / "gender").mkdir(exist_ok=True)
        pd.DataFrame(age_rows).to_csv(output_root / "age" / "labels.csv", index=False)
        pd.DataFrame(gender_rows).to_csv(
            output_root / "gender" / "labels.csv", index=False
        )

    summary = {
        "fairface_age": len(age_rows),
        "gender": len(gender_rows),
        "assignment_gender": len(assignment_rows),
    }
    print(summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fairface-root", type=Path, default=DEFAULT_FAIRFACE_ROOT)
    parser.add_argument(
        "--assignment-archive", type=Path, default=DEFAULT_ASSIGNMENT_ARCHIVE
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


if __name__ == "__main__":
    prepare(**vars(parse_args()))
