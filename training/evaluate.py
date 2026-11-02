"""Evaluate the trained model (headline metric is macro-F1).

Handles the clean PlantVillage test split and the PlantDoc field set, which has a
different subset of classes. Everything is scored in the model's label space
(models/labels.json): ground-truth folder names are mapped back by name, samples
from classes the model never learned are skipped, and macro-F1 is averaged over
the classes actually present. Metrics go to models/eval_report*.json.

    python -m training.evaluate                                 # PlantVillage test
    python -m training.evaluate --data data/processed_plantdoc   # field test set
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from training.config import TrainConfig
from training.dataset import load_split


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="path to .keras model")
    parser.add_argument("--data", default=None, help="processed data dir (with test/)")
    args = parser.parse_args()

    cfg = TrainConfig()
    model_path = Path(args.model) if args.model else cfg.output_dir / "kisanai_efficientnet.keras"
    data_dir = Path(args.data) if args.data else cfg.data_dir

    # model label space: index -> class name
    train_labels = json.loads((cfg.output_dir / "labels.json").read_text(encoding="utf-8"))
    name_to_idx = {name: i for i, name in enumerate(train_labels)}
    model = tf.keras.models.load_model(model_path)

    test_ds, eval_class_names = load_split(data_dir / "test", cfg.image_size, cfg.batch_size, False)

    y_true, y_pred, top3_hits, skipped = [], [], [], 0
    for images, batch_labels in test_ds:
        probs = model.predict(images, verbose=0)
        preds = np.argmax(probs, axis=1)
        top3 = np.argsort(probs, axis=1)[:, -3:]  # top-3 class indices per sample
        for true_local, pred_idx, t3 in zip(
            batch_labels.numpy().tolist(), preds.tolist(), top3.tolist()
        ):
            true_name = eval_class_names[true_local]
            if true_name not in name_to_idx:
                skipped += 1  # a class the model was never trained on
                continue
            ti = name_to_idx[true_name]
            y_true.append(ti)
            y_pred.append(int(pred_idx))
            top3_hits.append(1 if ti in t3 else 0)

    y_true, y_pred = np.array(y_true), np.array(y_pred)
    if len(y_true) == 0:
        raise SystemExit("No evaluable samples (no overlapping classes with the model).")
    if skipped:
        print(f"(skipped {skipped} samples from classes not in the model's label set)")

    all_idx = list(range(len(train_labels)))
    present = sorted(set(y_true.tolist()))  # classes actually in this test set

    accuracy = accuracy_score(y_true, y_pred)
    top3_accuracy = float(np.mean(top3_hits))
    macro_precision = precision_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    macro_recall = recall_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, labels=present, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, labels=present, average="weighted", zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, labels=all_idx, average=None, zero_division=0)

    print("\n================ EVALUATION ================")
    print(f"Data           : {data_dir}")
    print(f"Samples        : {len(y_true)}   |   classes present: {len(present)}/{len(train_labels)}")
    print(f"Accuracy       : {accuracy:.4f}")
    print(f"Top-3 accuracy : {top3_accuracy:.4f}")
    print(f"Macro precision: {macro_precision:.4f}")
    print(f"Macro recall   : {macro_recall:.4f}")
    print(f"Macro-F1       : {macro_f1:.4f}   <-- headline metric (over present classes)")
    print(f"Weighted-F1    : {weighted_f1:.4f}")
    print("\nPer-class report (present classes):")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=present,
            target_names=[train_labels[i] for i in present],
            digits=4,
            zero_division=0,
        )
    )

    report = {
        "data_dir": str(data_dir),
        "accuracy": float(accuracy),
        "top3_accuracy": top3_accuracy,
        "macro_precision": float(macro_precision),
        "macro_recall": float(macro_recall),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "num_samples": int(len(y_true)),
        "num_classes_present": len(present),
        "per_class_f1": {train_labels[i]: float(per_class_f1[i]) for i in present},
    }
    tag = "_plantdoc" if "plantdoc" in str(data_dir).lower() else ""
    out_json = cfg.output_dir / f"eval_report{tag}.json"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n✅ Saved metrics → {out_json}")

    # Confusion matrix over present classes.
    try:
        import matplotlib.pyplot as plt

        cm = confusion_matrix(y_true, y_pred, labels=present)
        names = [train_labels[i] for i in present]
        fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.5),) * 2)
        ConfusionMatrixDisplay(cm, display_labels=names).plot(
            ax=ax, xticks_rotation="vertical", colorbar=False
        )
        plt.tight_layout()
        cm_path = cfg.output_dir / f"confusion_matrix{tag}.png"
        fig.savefig(cm_path, dpi=120)
        print(f"✅ Saved confusion matrix → {cm_path}")
    except Exception as exc:  # pragma: no cover
        print(f"(Skipped confusion matrix plot: {exc})")


if __name__ == "__main__":
    main()
