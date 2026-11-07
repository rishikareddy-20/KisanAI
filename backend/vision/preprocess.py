"""Image preprocessing for the vision model. Must mirror training (training/dataset.py)."""
from __future__ import annotations

import io
from typing import Tuple

import numpy as np
from PIL import Image

DEFAULT_INPUT_SIZE = (224, 224)


def load_image(image_bytes: bytes) -> Image.Image:
    """Decode raw bytes into an RGB PIL image."""
    return Image.open(io.BytesIO(image_bytes)).convert("RGB")


def preprocess(
    image: Image.Image,
    input_size: Tuple[int, int] = DEFAULT_INPUT_SIZE,
) -> np.ndarray:
    """Resize and normalize into a batch of shape (1, H, W, 3)."""
    image = image.resize(input_size, Image.BILINEAR)
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.expand_dims(array, axis=0)
