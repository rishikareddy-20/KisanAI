"""Benchmark the exported model: Keras vs TF-Lite size, param count, and CPU
latency per image. Writes models/benchmark_report.json.

    python -m training.benchmark
    python -m training.benchmark --runs 100
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from training.config import TrainConfig


def _size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024) if path.exists() else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50, help="timed inference runs")
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()

    cfg = TrainConfig()
    keras_path = cfg.output_dir / "kisanai_efficientnet.keras"
    tflite_path = cfg.output_dir / "kisanai_efficientnet.tflite"

    report: dict = {}

    # ── Sizes ───────────────────────────────────────────────
    keras_mb = _size_mb(keras_path)
    tflite_mb = _size_mb(tflite_path)
    report["keras_size_mb"] = round(keras_mb, 3)
    report["tflite_size_mb"] = round(tflite_mb, 3)
    if keras_mb and tflite_mb:
        report["compression_ratio"] = round(keras_mb / tflite_mb, 2)

    # ── Param count (Keras) ─────────────────────────────────
    try:
        import tensorflow as tf

        model = tf.keras.models.load_model(keras_path)
        report["total_params"] = int(model.count_params())
        report["trainable_params"] = int(
            sum(np.prod(w.shape) for w in model.trainable_weights)
        )
    except Exception as exc:  # pragma: no cover
        print(f"(Skipped param count: {exc})")

    # ── TF-Lite latency ─────────────────────────────────────
    if tflite_path.exists():
        try:
            from tflite_runtime.interpreter import Interpreter  # type: ignore
        except ImportError:
            from tensorflow.lite import Interpreter  # type: ignore

        interp = Interpreter(model_path=str(tflite_path))
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]
        shape = inp["shape"]
        dummy = np.random.rand(*shape).astype(inp["dtype"] if inp["dtype"] != np.uint8 else np.uint8)
        if inp["dtype"] == np.uint8:
            dummy = (np.random.rand(*shape) * 255).astype(np.uint8)

        for _ in range(args.warmup):
            interp.set_tensor(inp["index"], dummy)
            interp.invoke()

        times = []
        for _ in range(args.runs):
            t0 = time.perf_counter()
            interp.set_tensor(inp["index"], dummy)
            interp.invoke()
            times.append((time.perf_counter() - t0) * 1000.0)  # ms

        times = np.array(times)
        report["tflite_latency_ms"] = {
            "mean": round(float(times.mean()), 2),
            "p50": round(float(np.percentile(times, 50)), 2),
            "p95": round(float(np.percentile(times, 95)), 2),
            "runs": args.runs,
        }
        report["tflite_throughput_img_per_s"] = round(1000.0 / times.mean(), 1)
    else:
        print(f"(No TF-Lite model at {tflite_path}; run training.export_tflite first)")

    out = cfg.output_dir / "benchmark_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n================ BENCHMARK ================")
    print(json.dumps(report, indent=2))
    print(f"\n✅ Saved → {out}")


if __name__ == "__main__":
    main()
