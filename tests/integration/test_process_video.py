import csv
from pathlib import Path

import cv2
import numpy as np

from process_video import process_video


def test_process_video_preserves_frame_count_and_csv_columns(tmp_path: Path) -> None:
    input_path = tmp_path / "input.mp4"
    output_path = tmp_path / "annotated.mp4"
    csv_path = tmp_path / "detections.csv"
    frame_rate = 10
    expected_frame_count = 30

    writer = cv2.VideoWriter(
        str(input_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        frame_rate,
        (64, 48),
    )
    assert writer.isOpened()
    for frame_index in range(expected_frame_count):
        frame = np.full(
            (48, 64, 3),
            frame_index % 255,
            dtype=np.uint8,
        )
        writer.write(frame)
    writer.release()

    class EmptyPipeline:
        def analyze(self, frame: np.ndarray) -> list[dict[str, object]]:
            return []

    process_video(input_path, output_path, csv_path, pipeline=EmptyPipeline())

    input_capture = cv2.VideoCapture(str(input_path))
    output_capture = cv2.VideoCapture(str(output_path))
    try:
        assert int(input_capture.get(cv2.CAP_PROP_FRAME_COUNT)) == expected_frame_count
        assert (
            int(output_capture.get(cv2.CAP_PROP_FRAME_COUNT))
            == expected_frame_count
        )
    finally:
        input_capture.release()
        output_capture.release()

    with csv_path.open(newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        assert reader.fieldnames == ["timestamp", "name", "gender", "age"]
