"""Transfer-learning model for approximate face-age regression."""

from __future__ import annotations

import math

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


INPUT_SHAPE = (128, 128, 3)
AGE_GROUPS = {
    "child": (0, 12),
    "teen": (13, 19),
    "young adult": (20, 35),
    "adult": (36, 59),
    "senior": (60, float("inf")),
}


def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-4,
    weights: str | None = "imagenet",
) -> keras.Model:
    """Build and compile a MobileNetV2 age-regression model.

    The model expects RGB images and returns one non-negative approximate age.
    The backbone is frozen initially so its layers can be fine-tuned later.
    """
    if input_shape[2:] != (3,):
        raise ValueError("input_shape must end with 3 channels for RGB images")
    if any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape dimensions must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    backbone = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights=weights,
    )
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="face")
    x = keras.applications.mobilenet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    x = layers.Dropout(0.30)(x)
    outputs = layers.Dense(
        1,
        activation="relu",
        name="age",
        kernel_initializer="he_normal",
    )(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="age_regressor")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.Huber(),
        metrics=[keras.metrics.MeanAbsoluteError(name="mae")],
    )
    return model


def predict_age_group(age: float) -> str:
    """Map a predicted age to child, teen, young adult, adult, or senior."""
    age = float(age)
    if not math.isfinite(age) or age < 0:
        raise ValueError("age must be a finite, non-negative number")

    for group, (minimum, maximum) in AGE_GROUPS.items():
        if minimum <= age <= maximum:
            return group
    raise ValueError(f"age is outside supported range: {age}")


__all__ = ["AGE_GROUPS", "INPUT_SHAPE", "build_model", "predict_age_group"]
