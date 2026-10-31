"""Benchmark baseline versus batched, cached face-pipeline inference."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from app.services.pipeline import FacePipeline


def load_images(directory: Path) -> list[np.ndarray]:
    images = []
    for path in sorted(directory.glob("*")):
        image = cv2.imread(str(path))
        if image is not None:
            images.append(image)
    if not images:
        raise ValueError(f"No readable images found in {directory}")
    return images


def benchmark(
    pipeline: FacePipeline,
    images: list[np.ndarray],
    repetitions: int,
) -> dict[str, float]:
    elapsed = 0.0
    faces = 0
    for _ in range(repetitions):
        for image in images:
            start = time.perf_counter()
            results = pipeline.analyze(image)
            elapsed += time.perf_counter() - start
            faces += len(results)
    image_count = len(images) * repetitions
    return {
        "average_ms_per_image": elapsed * 1000 / image_count,
        "average_ms_per_face": elapsed * 1000 / faces if faces else 0.0,
        "images": float(image_count),
        "faces": float(faces),
    }


def print_result(label: str, result: dict[str, float]) -> None:
    print(
        f"{label}: {result['average_ms_per_image']:.2f} ms/image, "
        f"{result['average_ms_per_face']:.2f} ms/face "
        f"({int(result['images'])} images, {int(result['faces'])} faces)"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions <= 0:
        raise ValueError("--repetitions must be positive")

    images = load_images(args.image_dir)
    baseline = FacePipeline(optimized=False)
    optimized = FacePipeline(optimized=True)
    before = benchmark(baseline, images, args.repetitions)
    after = benchmark(optimized, images, args.repetitions)
    print_result("Before (per-face inference)", before)
    print_result("After (batched inference + cache)", after)
    print(
        f"Speedup: {before['average_ms_per_image'] / after['average_ms_per_image']:.2f}x"
    )


if __name__ == "__main__":
    main()
