"""
Create (train, val, test) datasets for the *current* directory layout:

horse_images/data/train/*.jpg   → label 0
truck_images/data/train/*.jpg   → label 1
(and the same for val/ and test/)

No sub-folders per class are required, so image_dataset_from_directory()
is replaced by a manual tf.data pipeline.
"""
from __future__ import annotations
from pathlib import Path
from typing import Tuple

import tensorflow as tf
import config


class HorseTruckData:
    """Factory object that yields the three datasets."""

    def __init__(
        self,
        horse_root: Path,
        truck_root: Path,
        image_size: Tuple[int, int] = config.IMAGE_SIZE,
        batch_size: int = config.BATCH_SIZE,
        seed: int = config.SEED,
    ) -> None:
        self.horse_root = horse_root
        self.truck_root = truck_root
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed

    # ---------------------------------------------------------------- public
    def load(self) -> tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset]:
        """Return (train_ds, val_ds, test_ds)."""
        train = self._from_roots(
            self.horse_root / "train",
            self.truck_root / "train",
            shuffle=True,
        )
        val = self._from_roots(
            self.horse_root / "val",
            self.truck_root / "val",
            shuffle=False,
        )
        test = self._from_roots(
            self.horse_root / "test",
            self.truck_root / "test",
            shuffle=False,
        )
        return train, val, test

    # ---------------------------------------------------------------- private
    def _from_roots(
        self, horse_dir: Path, truck_dir: Path, *, shuffle: bool
    ) -> tf.data.Dataset:
        """Build one split by concatenating the two class datasets."""
        horse_ds = self._one_class_ds(horse_dir, label=0, shuffle=shuffle)
        truck_ds = self._one_class_ds(truck_dir, label=1, shuffle=shuffle)
        ds = horse_ds.concatenate(truck_ds)
        if shuffle:
            ds = ds.shuffle(1000, seed=self.seed)
        return ds.cache().prefetch(tf.data.AUTOTUNE)

    def _one_class_ds(
        self, dir_path: Path, *, label: int, shuffle: bool
    ) -> tf.data.Dataset:
        """
        Create a dataset for a single directory of JPEGs, attaching a constant
        binary label.
        """
        # 1) list all .jpg / .jpeg  (add png if you like)
        pattern = str(dir_path / "*.jpg")
        files = tf.data.Dataset.list_files(
            pattern, shuffle=shuffle, seed=self.seed
        )

        # 2) read + decode each file
        def _load(path: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
            img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
            img = tf.image.resize(img, self.image_size)
            img = tf.image.convert_image_dtype(img, tf.float32)  # 0-1 range
            return img, tf.cast(label, tf.float32)

        return (
            files.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(self.batch_size)
        )
