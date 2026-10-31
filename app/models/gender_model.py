"""Keras CNN for binary gender classification from face images."""

from __future__ import annotations

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


INPUT_SHAPE = (128, 128, 3)


def build_model(
    input_shape: tuple[int, int, int] = INPUT_SHAPE,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """Build and compile the gender classification CNN.

    The model expects RGB face images and returns one sigmoid probability:
    ``0`` represents male and ``1`` represents female.
    """
    if len(input_shape) != 3 or input_shape[2] != 3:
        raise ValueError("input_shape must be (height, width, 3) for RGB images")
    if any(dimension <= 0 for dimension in input_shape):
        raise ValueError("input_shape dimensions must be positive")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive")

    inputs = keras.Input(shape=input_shape, name="face")
    x = inputs

    for filters, dropout_rate in (
        (32, 0.20),
        (64, 0.25),
        (128, 0.30),
        (256, 0.35),
    ):
        x = layers.Conv2D(
            filters,
            kernel_size=3,
            padding="same",
            use_bias=False,
            kernel_initializer="he_normal",
        )(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)
        x = layers.MaxPooling2D(pool_size=2)(x)
        x = layers.Dropout(dropout_rate)(x)

    x = layers.GlobalAveragePooling2D(name="global_average_pooling")(x)
    outputs = layers.Dense(1, activation="sigmoid", name="gender")(x)
    model = keras.Model(inputs=inputs, outputs=outputs, name="gender_classifier")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss=keras.losses.BinaryCrossentropy(),
        metrics=[keras.metrics.BinaryAccuracy(name="accuracy")],
    )
    return model


__all__ = ["INPUT_SHAPE", "build_model"]
