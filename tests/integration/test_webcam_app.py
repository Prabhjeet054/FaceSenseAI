import numpy as np

import webcam_app


class FakeVideoCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = iter(frames)
        self.read_count = 0
        self.released = False

    def isOpened(self) -> bool:
        return True

    def read(self) -> tuple[bool, np.ndarray]:
        try:
            frame = next(self.frames)
        except StopIteration:
            return False, np.empty((0, 0, 3), dtype=np.uint8)
        self.read_count += 1
        return True, frame

    def release(self) -> None:
        self.released = True


class FakePipeline:
    def __init__(self) -> None:
        self.call_count = 0

    def analyze(self, frame: np.ndarray) -> list[dict[str, object]]:
        self.call_count += 1
        return []


def test_webcam_loop_processes_every_third_frame_and_releases_resources(
    monkeypatch,
) -> None:
    frames = [np.zeros((32, 32, 3), dtype=np.uint8) for _ in range(30)]
    capture = FakeVideoCapture(frames)
    pipeline = FakePipeline()
    wait_key_calls = 0

    monkeypatch.setattr(webcam_app.cv2, "VideoCapture", lambda index: capture)
    monkeypatch.setattr(webcam_app, "FacePipeline", lambda: pipeline)
    monkeypatch.setattr(webcam_app, "draw_faces", lambda frame, results: frame)
    monkeypatch.setattr(webcam_app.cv2, "imshow", lambda title, frame: None)

    def fake_wait_key(delay: int) -> int:
        nonlocal wait_key_calls
        wait_key_calls += 1
        return ord("q") if wait_key_calls == 30 else -1

    destroyed_windows = False

    def fake_destroy_windows() -> None:
        nonlocal destroyed_windows
        destroyed_windows = True

    monkeypatch.setattr(webcam_app.cv2, "waitKey", fake_wait_key)
    monkeypatch.setattr(webcam_app.cv2, "destroyAllWindows", fake_destroy_windows)

    webcam_app.run()

    assert capture.read_count == 30
    assert pipeline.call_count == 10
    assert capture.released
    assert destroyed_windows
