import numpy as np

from app.utils.draw import draw_face


def test_annotation_preserves_shape_and_changes_box_border_pixels() -> None:
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    result = {
        "box": [20, 30, 80, 60],
        "name": "Rahul",
        "gender": "male",
        "gender_confidence": 0.94,
        "age": 27,
    }

    annotated = draw_face(image, result)

    assert annotated.shape == image.shape
    assert not np.array_equal(annotated[30, 50], image[30, 50])
