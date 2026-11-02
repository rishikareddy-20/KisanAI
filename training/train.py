"""Train the crop-disease classifier (EfficientNet, two-phase transfer learning).

    python -m training.train            # full run
    python -m training.train --sample   # quick pass on a tiny subset

Writes a Keras model and labels.json into models/.
"""
from __future__ import annotations

import argparse
import json

import tensorflow as tf

from training.config import TrainConfig
from training.dataset import compute_class_weights, make_datasets


def build_model(cfg: TrainConfig, num_classes: int) -> tf.keras.Model:
    input_shape = (*cfg.image_size, 3)
    if cfg.backbone == "efficientnet_b3":
        base = tf.keras.applications.EfficientNetB3(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    else:
        base = tf.keras.applications.EfficientNetB0(
            include_top=False, weights="imagenet", input_shape=input_shape
        )
    base.trainable = False  # phase 1: freeze backbone

    inputs = tf.keras.Input(shape=input_shape)
    # EfficientNet wants [0, 255]; our pipeline feeds [0, 1]
    x = tf.keras.layers.Rescaling(255.0)(inputs)
    x = base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(cfg.dropout)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    model._base = base  # keep a handle for phase 2
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", action="store_true", help="quick sample run")
    parser.add_argument("--backbone", default=None, help="efficientnet_b0 | efficientnet_b3")
    args = parser.parse_args()

    cfg = TrainConfig()
    if args.backbone:
        cfg.backbone = args.backbone
    cfg.sample_run = args.sample
    cfg.apply_sample_overrides()
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, _test_ds, class_names = make_datasets(cfg)
    num_classes = len(class_names)
    print(f"Classes ({num_classes}): {class_names}")

    class_weights = (
        compute_class_weights(cfg.data_dir / "train", class_names)
        if cfg.use_class_weights
        else None
    )

    model = build_model(cfg, num_classes)

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            str(cfg.checkpoint_dir / "best.keras"), save_best_only=True, monitor="val_accuracy"
        ),
        tf.keras.callbacks.EarlyStopping(patience=4, restore_best_weights=True, monitor="val_loss"),
    ]

    # ── Phase 1: train the head with the backbone frozen ────
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.head_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== Phase 1: training classification head ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.head_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ── Phase 2: unfreeze top blocks and fine-tune ──────────
    base = model._base
    base.trainable = True
    for layer in base.layers[: -cfg.finetune_unfreeze_layers]:
        layer.trainable = False
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg.finetune_lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n=== Phase 2: fine-tuning top blocks ===")
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=cfg.finetune_epochs,
        class_weight=class_weights,
        callbacks=callbacks,
    )

    # ── Save artifacts ──────────────────────────────────────
    model_path = cfg.output_dir / "kisanai_efficientnet.keras"
    labels_path = cfg.output_dir / "labels.json"
    model.save(model_path)
    with labels_path.open("w", encoding="utf-8") as fh:
        json.dump(class_names, fh, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved model → {model_path}")
    print(f"✅ Saved labels → {labels_path}")
    print("Next: python -m training.evaluate   then   python -m training.export_tflite")


if __name__ == "__main__":
    main()
