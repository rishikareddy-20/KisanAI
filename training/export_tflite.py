"""Export the trained Keras model to a quantized TF-Lite artifact for the backend.

Usage:
    python -m training.export_tflite                 # INT8 dynamic-range quant
    python -m training.export_tflite --float16       # smaller, GPU-friendly
    python -m training.export_tflite --no-quant      # full float32

Writes models/kisanai_efficientnet.tflite, which the backend loads automatically.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from training.config import TrainConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None, help="path to .keras model")
    parser.add_argument("--float16", action="store_true", help="float16 quantization")
    parser.add_argument("--no-quant", action="store_true", help="no quantization (float32)")
    args = parser.parse_args()

    cfg = TrainConfig()
    model_path = Path(args.model) if args.model else cfg.output_dir / "kisanai_efficientnet.keras"
    model = tf.keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if not args.no_quant:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]  # dynamic-range INT8 by default
        if args.float16:
            converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()

    out_path = cfg.output_dir / "kisanai_efficientnet.tflite"
    out_path.write_bytes(tflite_model)
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"✅ Exported TF-Lite model → {out_path}  ({size_mb:.2f} MB)")
    print("The backend will pick this up automatically on next start.")


if __name__ == "__main__":
    main()
