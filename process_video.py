"""Analyze a video, save annotated frames, and export face detections to CSV."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import cv2

from app.services.pipeline import FacePipeline
from app.utils.draw import draw_faces


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_VIDEO = PROJECT_ROOT / "reports" / "annotated_video.mp4"
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "reports" / "detections.csv"
CSV_FIELDS = ("timestamp", "name", "gender", "age")


def process_video(
    input_path: str | Path,
    output_video_path: str | Path = DEFAULT_OUTPUT_VIDEO,
    output_csv_path: str | Path = DEFAULT_OUTPUT_CSV,
    pipeline: FacePipeline | Any | None = None,
) -> int:
    """Process every video frame and return the number of detection rows."""
    input_path = Path(input_path)
    output_video_path = Path(output_video_path)
    output_csv_path = Path(output_csv_path)
    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        capture.release()
        raise RuntimeError(f"Could not open input video: {input_path}")

    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    writer: cv2.VideoWriter | None = None
    detection_count = 0
    frame_index = 0
    analyzer = pipeline or FacePipeline()

    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if width <= 0 or height <= 0:
            raise RuntimeError("Input video has invalid dimensions")
        if fps <= 0:
            fps = 30.0

        writer = cv2.VideoWriter(
            str(output_video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Could not create output video: {output_video_path}")

        with output_csv_path.open("w", newline="", encoding="utf-8") as csv_file:
            csv_writer = csv.DictWriter(csv_file, fieldnames=CSV_FIELDS)
            csv_writer.writeheader()

            while True:
                success, frame = capture.read()
                if not success:
                    break

                results = analyzer.analyze(frame)
                writer.write(draw_faces(frame, results))
                timestamp = frame_index / fps
                for result in results:
                    csv_writer.writerow(
                        {
                            "timestamp": f"{timestamp:.3f}",
                            "name": result["name"],
                            "gender": result["gender"],
                            "age": result["age"],
                        }
                    )
                    detection_count += 1
                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    print(f"Processed {frame_index} frames and {detection_count} detections")
    print(f"Annotated video: {output_video_path}")
    print(f"Detection CSV: {output_csv_path}")
    return detection_count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_video", type=Path)
    parser.add_argument("--output-video", type=Path, default=DEFAULT_OUTPUT_VIDEO)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    process_video(**vars(arguments))
