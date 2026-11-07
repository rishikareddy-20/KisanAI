"""TF-Lite disease classifier.

Falls back to a deterministic mock when the .tflite model or interpreter is
missing, so the rest of the pipeline can run before a model is trained.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np

from .preprocess import DEFAULT_INPUT_SIZE, load_image, preprocess

logger = logging.getLogger("kisanai.vision")


@dataclass
class Prediction:
    disease: str                      # class key, e.g. "Tomato___Late_blight"
    confidence: float                 # top-1 probability [0, 1]
    uncertain: bool                   # confidence below threshold
    top_k: List[dict] = field(default_factory=list)
    is_mock: bool = False             # True when no trained model was loaded


def _load_labels(labels_path: Path) -> Optional[List[str]]:
    if labels_path.exists():
        with labels_path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Accept a plain list, {"classes": [...]}, or a dict keyed by index.
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            if "classes" in data:
                return list(data["classes"])
            return [data[k] for k in sorted(data, key=lambda x: int(x))]
    return None


def _try_load_interpreter(model_path: Path):
    """Load a TF-Lite interpreter from whichever runtime is installed."""
    if not model_path.exists():
        logger.warning("TF-Lite model %s not found; using mock classifier.", model_path)
        return None

    try:
        from tflite_runtime.interpreter import Interpreter
    except ImportError:
        try:
            from tensorflow.lite.python.interpreter import Interpreter
        except ImportError:
            logger.exception("No compatible TensorFlow Lite interpreter found.")
            return None

    try:
        interpreter = Interpreter(model_path=str(model_path))
        interpreter.allocate_tensors()
        logger.info("Loaded TensorFlow Lite model: %s", model_path)
        return interpreter
    except Exception:
        logger.exception("Failed to load TensorFlow Lite model: %s", model_path)
        return None

class DiseaseClassifier:
    """Loads the TF-Lite model once and classifies leaf images."""

    def __init__(
        self,
        model_path: Path,
        labels_path: Path,
        confidence_threshold: float = 0.55,
        top_k: int = 3,
    ) -> None:
        self.confidence_threshold = confidence_threshold
        self.top_k = top_k
        self.labels = _load_labels(labels_path) or _FALLBACK_LABELS
        self.interpreter = _try_load_interpreter(model_path)
        self.is_mock = self.interpreter is None

        if not self.is_mock:
            self._in = self.interpreter.get_input_details()[0]
            self._out = self.interpreter.get_output_details()[0]
            shape = self._in["shape"]
            self.input_size = (int(shape[1]), int(shape[2]))
        else:
            self.input_size = DEFAULT_INPUT_SIZE

    def predict(self, image_bytes: bytes) -> Prediction:
        image = load_image(image_bytes)
        if self.is_mock:
            probs = self._mock_probs(image_bytes)
        else:
            probs = self._infer(image)
        return self._to_prediction(probs)

    def _infer(self, image) -> np.ndarray:
        batch = preprocess(image, self.input_size)
        dtype = self._in["dtype"]
        if dtype == np.uint8:  # quantized model
            scale, zero_point = self._in.get("quantization", (0.0, 0))
            if scale:
                batch = (batch / scale + zero_point).astype(dtype)
            else:
                batch = (batch * 255).astype(dtype)
        self.interpreter.set_tensor(self._in["index"], batch)
        self.interpreter.invoke()
        out = self.interpreter.get_tensor(self._out["index"])[0]
        if self._out["dtype"] == np.uint8:
            scale, zero_point = self._out.get("quantization", (0.0, 0))
            out = (out.astype(np.float32) - zero_point) * (scale or 1.0)
        out = out.astype(np.float32)
        # The trained model already ends in softmax, so don't softmax again
        # (that would flatten the confidence). Only softmax raw-logits outputs.
        if np.all(out >= 0) and abs(float(out.sum()) - 1.0) < 1e-2:
            return out / out.sum()  # already probabilities, just renormalise
        return self._softmax(out)

    def _mock_probs(self, image_bytes: bytes) -> np.ndarray:
        """Deterministic pseudo-prediction seeded by image content (same image, same result)."""
        seed = int(hashlib.sha256(image_bytes).hexdigest(), 16) % (2**32)
        rng = np.random.default_rng(seed)
        logits = rng.normal(size=len(self.labels))
        # Bias one class so the confidence looks realistic.
        logits[rng.integers(len(self.labels))] += 4.0
        return self._softmax(logits)

    def _to_prediction(self, probs: np.ndarray) -> Prediction:
        order = np.argsort(probs)[::-1]
        top = order[: self.top_k]
        top_k = [
            {"disease": self.labels[i], "confidence": round(float(probs[i]), 4)}
            for i in top
        ]
        best = int(order[0])
        confidence = float(probs[best])
        return Prediction(
            disease=self.labels[best],
            confidence=round(confidence, 4),
            uncertain=confidence < self.confidence_threshold,
            top_k=top_k,
            is_mock=self.is_mock,
        )

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        e = np.exp(x - np.max(x))
        return e / e.sum()


# Fallback labels (subset of the 38 PlantVillage classes), used only when
# models/labels.json is missing.
_FALLBACK_LABELS = [
    "Apple___Apple_scab",
    "Apple___Black_rot",
    "Apple___healthy",
    "Corn___Common_rust",
    "Corn___Northern_Leaf_Blight",
    "Corn___healthy",
    "Grape___Black_rot",
    "Grape___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Rice___Brown_spot",
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Yellow_Leaf_Curl_Virus",
    "Tomato___healthy",
]
