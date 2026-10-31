from pathlib import Path

import pytest

cv2 = pytest.importorskip("cv2")

from app.core.detector import FaceDetector
from config import FACE_DETECTION_MODEL_PATH


FIXTURES_DIR = Path(__file__).parent / "fixtures" / "faces"
SAMPLE_IMAGES = (
    ("one_face.jpg", 1),
    ("no_face.jpg", 0),
    ("group.jpg", 3),
)


@pytest.fixture(scope="module")
def face_detector() -> FaceDetector:
    if not FACE_DETECTION_MODEL_PATH.is_file():
        pytest.skip(
            f"YuNet model not found at {FACE_DETECTION_MODEL_PATH}; "
            "download it before running detector integration tests"
        )
    return FaceDetector()


@pytest.mark.parametrize(("filename", "expected_count"), SAMPLE_IMAGES)
def test_detects_expected_number_of_faces(
    face_detector: FaceDetector, filename: str, expected_count: int
) -> None:
    image_path = FIXTURES_DIR / filename
    if not image_path.is_file():
        pytest.skip(f"Sample image not found: {image_path}")

    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read sample image: {image_path}"

    detections = face_detector.detect(image)

    assert len(detections) == expected_count


@pytest.mark.parametrize(("filename", "_expected_count"), SAMPLE_IMAGES)
def test_detected_boxes_are_inside_image(
    face_detector: FaceDetector, filename: str, _expected_count: int
) -> None:
    image_path = FIXTURES_DIR / filename
    if not image_path.is_file():
        pytest.skip(f"Sample image not found: {image_path}")

    image = cv2.imread(str(image_path))
    assert image is not None, f"Could not read sample image: {image_path}"
    height, width = image.shape[:2]

    for detection in face_detector.detect(image):
        x, y, box_width, box_height = detection["bbox"]
        assert 0 <= x <= width
        assert 0 <= y <= height
        assert 0 <= box_width <= width
        assert 0 <= box_height <= height
        assert x + box_width <= width
        assert y + box_height <= height
