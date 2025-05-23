"""
python main.py
--------------
Creates the datasets, builds the model, trains, evaluates and prints metrics.
"""
from pathlib import Path
import config
from data import HorseTruckData
from model import HorseTruckCNN
from trainer import Trainer


def main() -> None:
    # ---------------------------------------------------------------- data
    data_loader = HorseTruckData(config.HORSE_DATA_DIR,
                                 config.TRUCK_DATA_DIR)
    train_ds, val_ds, test_ds = data_loader.load()

    # ---------------------------------------------------------------- model
    model = HorseTruckCNN.build()
    model.summary()

    # ---------------------------------------------------------------- train
    trainer = Trainer(model,
                      checkpoint_dir=Path("./checkpoints"))
    history = trainer.fit(train_ds, val_ds)

    # ---------------------------------------------------------------- test
    loss, acc = trainer.evaluate(test_ds)
    print(f"\n✅  Test accuracy = {acc:.4f}")

    # optional: plot training curves
    _plot_history(history)


def _plot_history(history) -> None:
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 3))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="val")
    plt.title("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["accuracy"], label="train")
    plt.plot(history.history["val_accuracy"], label="val")
    plt.title("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
