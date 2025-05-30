"""
Early-stopping & optional checkpointing.
"""
from pathlib import Path
from typing import Optional, Tuple
import tensorflow as tf
import config


class Trainer:
    def __init__(self,
                 model: tf.keras.Model,
                 lr: float = config.LEARNING_RATE,
                 early_stop: int = config.EARLY_STOP,
                 checkpoint_dir: Optional[Path] = None):
        self.model = model
        self.lr = lr
        self.early_stop = early_stop
        self.checkpoint_dir = checkpoint_dir

    # --------------------------------------------------------------  public
    def fit(self,
            train_ds: tf.data.Dataset,
            val_ds: tf.data.Dataset,
            epochs: int = config.EPOCHS) -> tf.keras.callbacks.History:
        self._compile()

        callbacks = [tf.keras.callbacks.EarlyStopping(
                        monitor="val_loss",
                        patience=self.early_stop,
                        restore_best_weights=True)]
        if self.checkpoint_dir:
            callbacks.append(
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=str(self.checkpoint_dir / "best.h5"),
                    save_best_only=True,
                    monitor="val_loss"))

        return self.model.fit(train_ds,
                              validation_data=val_ds,
                              epochs=epochs,
                              callbacks=callbacks)

    def evaluate(self,
                 test_ds: tf.data.Dataset) -> Tuple[float, float]:
        """Returns (loss, accuracy)."""
        return self.model.evaluate(test_ds, verbose=0)

    # -------------------------------------------------------------- internal
    def _compile(self) -> None:
        self.model.compile(optimizer=tf.keras.optimizers.Adam(self.lr),
                           loss="binary_crossentropy",
                           metrics=["accuracy"])
