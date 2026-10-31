import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

tf = pytest.importorskip("tensorflow")

import evaluate


class FakeModel:
    def __init__(self, predictions: list[float]) -> None:
        self.predictions = np.asarray(predictions, dtype=np.float32).reshape(-1, 1)

    def predict(self, images: np.ndarray, verbose: int = 0) -> np.ndarray:
        assert len(images) == len(self.predictions)
        return self.predictions


def test_evaluate_writes_metrics_json_for_tiny_dataset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "processed"
    test_dir = data_dir / "test"
    test_dir.mkdir(parents=True)
    records = []
    for index, (gender, age) in enumerate(((0, 10), (1, 25), (0, 45), (1, 70))):
        filename = f"sample_{index}.png"
        image = tf.zeros((128, 128, 3), dtype=tf.uint8)
        tf.io.write_file(
            str(test_dir / filename),
            tf.io.encode_png(image),
        )
        records.append(
            {"filename": filename, "age": age, "gender": gender, "split": "test"}
        )
    pd.DataFrame(records).to_csv(data_dir / "labels.csv", index=False)

    gender_model_path = tmp_path / "gender.h5"
    age_model_path = tmp_path / "age.h5"
    gender_model_path.touch()
    age_model_path.touch()
    models = iter(
        [
            FakeModel([0.1, 0.9, 0.2, 0.8]),
            FakeModel([10.0, 25.0, 45.0, 70.0]),
        ]
    )
    monkeypatch.setattr(
        evaluate.tf.keras.models,
        "load_model",
        lambda path, compile=False: next(models),
    )

    report_path = tmp_path / "reports" / "metrics.json"
    evaluate.evaluate(
        data_dir=data_dir,
        gender_model_path=gender_model_path,
        age_model_path=age_model_path,
        report_path=report_path,
    )

    assert report_path.is_file()
    metrics = json.loads(report_path.read_text())
    assert "gender_accuracy" in metrics
    assert "age_mae" in metrics
    assert metrics["gender_accuracy"] == 1.0
    assert metrics["age_mae"] == 0.0
