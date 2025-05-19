"""
Thin wrapper that glues data and model together and runs training / evaluation.
"""
import matplotlib.pyplot as plt
import tensorflow as tf

from . import config as C
from .data import HorseTruckDataModule
from .model import CNNHorseTruck


class Trainer:
    def __init__(self):
        self.data_module = HorseTruckDataModule()
        (self.train_ds,
         self.val_ds,
         self.test_ds) = self.data_module.load()

        self.model = CNNHorseTruck.compile_model()

    # -----------------------------------------------------------------
    def fit(self):
        early_stop = tf.keras.callbacks.EarlyStopping(
            patience=C.EARLY_STOP,
            restore_best_weights=True
        )

        history = self.model.fit(
            self.train_ds,
            validation_data=self.val_ds,
            epochs=C.EPOCHS,
            callbacks=[early_stop],
            verbose=2
        )

        self._plot_learning_curves(history)

    # -----------------------------------------------------------------
    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_ds, verbose=0)
        print(f"🔎  Test accuracy: {test_acc:0.3f}   |   Test loss: {test_loss:0.4f}")

    # -----------------------------------------------------------------
    def predict_sample_batch(self, n=9):
        import numpy as np
        plt.figure(figsize=(8, 8))

        images, labels = next(iter(self.test_ds))
        preds = self.model.predict(images, verbose=0).squeeze()
        preds = (preds > 0.5).astype(int)

        class_names = self.data_module.class_names

        for i in range(n):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            true_lbl  = class_names[labels[i]]
            pred_lbl  = class_names[preds[i]]
            plt.title(f"Truth: {true_lbl}\nPred : {pred_lbl}",
                      color="green" if true_lbl == pred_lbl else "red",
                      fontsize=8)
            plt.axis("off")
        plt.tight_layout()
        plt.show()

    # -----------------------------------------------------------------
    @staticmethod
    def _plot_learning_curves(history):
        import matplotlib.pyplot as plt
        acc      = history.history["accuracy"]
        val_acc  = history.history["val_accuracy"]
        loss     = history.history["loss"]
        val_loss = history.history["val_loss"]
        epochs   = range(1, len(acc) + 1)

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(epochs, acc, label="train")
        plt.plot(epochs, val_acc, label="val")
        plt.title("Accuracy")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs, loss, label="train")
        plt.plot(epochs, val_loss, label="val")
        plt.title("Loss")
        plt.legend()
        plt.show()
