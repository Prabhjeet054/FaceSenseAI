"""Run FaceSense AI face analysis on a webcam stream."""

from __future__ import annotations

import argparse
import time
from collections import deque
from typing import Any

import cv2
import numpy as np

from app.models.age_model import predict_age_group
from app.services.pipeline import FacePipeline
from app.utils.draw import draw_faces


DEFAULT_CAMERA_INDEX = 0
DEFAULT_PROCESS_EVERY = 3
DEFAULT_AGE_WINDOW = 5
IOU_MATCH_THRESHOLD = 0.30


def box_iou(first: list[float], second: list[float]) -> float:
    """Return intersection-over-union for ``x, y, width, height`` boxes."""
    first_x1, first_y1 = first[:2]
    first_x2 = first_x1 + first[2]
    first_y2 = first_y1 + first[3]
    second_x1, second_y1 = second[:2]
    second_x2 = second_x1 + second[2]
    second_y2 = second_y1 + second[3]

    intersection_width = max(0.0, min(first_x2, second_x2) - max(first_x1, second_x1))
    intersection_height = max(0.0, min(first_y2, second_y2) - max(first_y1, second_y1))
    intersection = intersection_width * intersection_height
    union = first[2] * first[3] + second[2] * second[3] - intersection
    return intersection / union if union > 0 else 0.0


class AgeSmoother:
    """Maintain moving-average ages for faces matched by bounding-box overlap."""

    def __init__(self, window_size: int = DEFAULT_AGE_WINDOW) -> None:
        if window_size <= 0:
            raise ValueError("window_size must be positive")
        self.window_size = window_size
        self.tracks: dict[int, dict[str, Any]] = {}
        self.next_track_id = 0

    def smooth(self, results: list[dict[str, Any]]) -> list[dict[str, Any]]:
        updated_tracks: dict[int, dict[str, Any]] = {}
        unmatched_tracks = set(self.tracks)
        smoothed_results: list[dict[str, Any]] = []

        for result in results:
            best_track_id = None
            best_iou = IOU_MATCH_THRESHOLD
            for track_id in unmatched_tracks:
                overlap = box_iou(result["box"], self.tracks[track_id]["box"])
                if overlap > best_iou:
                    best_track_id = track_id
                    best_iou = overlap

            if best_track_id is None:
                best_track_id = self.next_track_id
                self.next_track_id += 1
                ages: deque[float] = deque(maxlen=self.window_size)
            else:
                unmatched_tracks.remove(best_track_id)
                ages = self.tracks[best_track_id]["ages"]

            ages.append(float(result["age"]))
            smoothed_result = dict(result)
            smoothed_result["age"] = sum(ages) / len(ages)
            smoothed_result["age_group"] = predict_age_group(smoothed_result["age"])
            updated_tracks[best_track_id] = {
                "box": result["box"],
                "ages": ages,
            }
            result = smoothed_result
            result["_track_id"] = best_track_id
            smoothed_results.append(result)

        self.tracks = updated_tracks
        return smoothed_results


def run(
    camera_index: int = DEFAULT_CAMERA_INDEX,
    process_every: int = DEFAULT_PROCESS_EVERY,
    age_window: int = DEFAULT_AGE_WINDOW,
) -> None:
    """Open a webcam and display live face analysis until ``q`` is pressed."""
    if process_every <= 0:
        raise ValueError("process_every must be positive")
    pipeline = FacePipeline()
    smoother = AgeSmoother(age_window)
    camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        camera.release()
        raise RuntimeError(f"Could not open webcam with camera index {camera_index}")

    frame_number = 0
    last_results: list[dict[str, Any]] = []
    last_time = time.perf_counter()
    fps = 0.0
    try:
        while True:
            success, frame = camera.read()
            if not success:
                raise RuntimeError("Could not read a frame from the webcam")

            frame_number += 1
            if frame_number % process_every == 1:
                last_results = smoother.smooth(pipeline.analyze(frame))

            now = time.perf_counter()
            elapsed = now - last_time
            if elapsed > 0:
                fps = 1.0 / elapsed
            last_time = now
            annotated = draw_faces(frame, last_results)
            cv2.putText(
                annotated,
                f"FPS: {fps:.1f}",
                (12, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("FaceSense AI - press q to quit", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        camera.release()
        cv2.destroyAllWindows()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera-index", type=int, default=DEFAULT_CAMERA_INDEX)
    parser.add_argument("--process-every", type=int, default=DEFAULT_PROCESS_EVERY)
    parser.add_argument("--age-window", type=int, default=DEFAULT_AGE_WINDOW)
    return parser.parse_args()


if __name__ == "__main__":
    run(**vars(parse_args()))
