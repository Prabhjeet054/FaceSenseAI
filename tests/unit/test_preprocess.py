import cv2
import numpy as np

from app.core.preprocess import align_face


def _rotate_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack((points, np.ones(len(points))))
    return homogeneous @ matrix.T


def test_align_face_makes_twenty_degree_eye_line_horizontal() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    eyes = np.array([[50.0, 60.0], [110.0, 60.0]], dtype=np.float32)
    for x, y in eyes.astype(int):
        cv2.circle(image, (x, y), 5, (255, 255, 255), -1)

    rotation = cv2.getRotationMatrix2D((80, 60), 20, 1.0)
    rotated_image = cv2.warpAffine(image, rotation, (160, 120))
    rotated_eyes = _rotate_points(eyes, rotation)

    aligned_image = align_face(rotated_image, rotated_eyes)
    mask = cv2.inRange(
        aligned_image,
        np.array([200, 200, 200], dtype=np.uint8),
        np.array([255, 255, 255], dtype=np.uint8),
    )
    _, _, _, centers = cv2.connectedComponentsWithStats(mask)
    aligned_eyes = centers[1:]
    assert len(aligned_eyes) == 2

    eye_line_angle = np.degrees(
        np.arctan2(
            aligned_eyes[1, 1] - aligned_eyes[0, 1],
            aligned_eyes[1, 0] - aligned_eyes[0, 0],
        )
    )
    assert abs(eye_line_angle) <= 2
