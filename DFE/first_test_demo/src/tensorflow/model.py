"""
CNN = 3×(Conv→BN→ReLU→MaxPool) → Flatten → Dense(128) → Dropout → Sigmoid
"""

from typing import Tuple
import tensorflow as tf
import config


class HorseTruckCNN:
    """Builds the Keras model according to hyper-parameters in config.py"""

    @staticmethod
    def build(input_shape: Tuple[int, int, int] = (*config.IMAGE_SIZE, 3),
              num_filters: Tuple[int, ...] = config.NUM_FILTERS,
              kernel_size: int = config.KERNEL_SIZE,
              dense_units: int = config.DENSE_UNITS,
              dropout: float = config.DROPOUT) -> tf.keras.Model:
        inputs = tf.keras.layers.Input(shape=input_shape)

        # Rescale 0-255 → 0-1 (faster convergence, see Scaling PDF [[3]])
        x = tf.keras.layers.Rescaling(1.0 / 255)(inputs)

        # --- convolutional backbone
        for f in num_filters:
            x = tf.keras.layers.Conv2D(f, kernel_size,
                                       padding="same",
                                       activation="relu")(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.MaxPooling2D()(x)

        # --- head
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(dense_units, activation="relu")(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dropout(dropout)(x)
        outputs = tf.keras.layers.Dense(1, activation="sigmoid")(x)

        return tf.keras.Model(inputs, outputs, name="HorseTruckCNN")
