"""Dataset loading and augmentation.

Expects the ImageFolder layout from data/prepare.py (train/val/test, one
subfolder per class). Augmentation adds occlusion/noise-like transforms to
help the model carry over from clean PlantVillage to PlantDoc field photos.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import tensorflow as tf

AUTOTUNE = tf.data.AUTOTUNE


def _augmentation() -> tf.keras.Sequential:
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip("horizontal_and_vertical"),
            tf.keras.layers.RandomRotation(0.2),
            tf.keras.layers.RandomZoom(0.2),
            tf.keras.layers.RandomContrast(0.2),
            # value_range must match our [0, 1] pipeline (Keras default is 0-255).
            tf.keras.layers.RandomBrightness(0.2, value_range=(0.0, 1.0)),
            # stands in for partial occlusion / off-centre field shots
            tf.keras.layers.RandomTranslation(0.1, 0.1),
        ],
        name="augmentation",
    )


def load_split(
    split_dir: Path,
    image_size: Tuple[int, int],
    batch_size: int,
    shuffle: bool,
) -> Tuple[tf.data.Dataset, List[str]]:
    ds = tf.keras.utils.image_dataset_from_directory(
        str(split_dir),
        image_size=image_size,
        batch_size=batch_size,
        shuffle=shuffle,
        label_mode="int",
    )
    class_names = ds.class_names
    # scale to [0, 1] to match backend/vision/preprocess.py
    ds = ds.map(lambda x, y: (x / 255.0, y), num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE), class_names


def make_datasets(cfg) -> Tuple[tf.data.Dataset, tf.data.Dataset, tf.data.Dataset, List[str]]:
    train_ds, class_names = load_split(cfg.data_dir / "train", cfg.image_size, cfg.batch_size, True)
    val_ds, _ = load_split(cfg.data_dir / "val", cfg.image_size, cfg.batch_size, False)
    test_ds, _ = load_split(cfg.data_dir / "test", cfg.image_size, cfg.batch_size, False)

    augment = _augmentation()
    # clip after augmentation so brightness/contrast jitter stays in [0, 1]
    train_ds = train_ds.map(
        lambda x, y: (tf.clip_by_value(augment(x, training=True), 0.0, 1.0), y),
        num_parallel_calls=AUTOTUNE,
    )
    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_dir: Path, class_names: List[str]) -> dict:
    """Inverse-frequency class weights to counter PlantVillage imbalance."""
    counts = []
    for name in class_names:
        counts.append(len(list((train_dir / name).glob("*"))))
    total = sum(counts) or 1
    n_classes = len(class_names)
    return {
        i: total / (n_classes * max(c, 1))
        for i, c in enumerate(counts)
    }
