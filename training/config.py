"""Training config. sample_run does a quick pass to check the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

ROOT_DIR = Path(__file__).resolve().parent.parent


@dataclass
class TrainConfig:
    # Data
    data_dir: Path = ROOT_DIR / "data" / "processed"   # expects train/ val/ test/ subfolders
    image_size: Tuple[int, int] = (224, 224)
    batch_size: int = 32

    # Model
    backbone: str = "efficientnet_b0"  # efficientnet_b0 | efficientnet_b3
    dropout: float = 0.3

    # Two-phase transfer learning
    head_epochs: int = 5          # phase 1: frozen backbone
    finetune_epochs: int = 15     # phase 2: unfreeze top blocks
    head_lr: float = 1e-3
    finetune_lr: float = 1e-5
    finetune_unfreeze_layers: int = 30  # number of top layers to unfreeze

    # Imbalance handling
    use_class_weights: bool = True

    # Output
    output_dir: Path = ROOT_DIR / "models"
    checkpoint_dir: Path = ROOT_DIR / "models" / "checkpoints"

    # Quick sample run (overrides for a fast end-to-end sanity check)
    sample_run: bool = False
    sample_limit_per_class: int = 20
    sample_epochs: int = 1

    def apply_sample_overrides(self) -> None:
        if self.sample_run:
            self.head_epochs = self.sample_epochs
            self.finetune_epochs = self.sample_epochs
            self.batch_size = min(self.batch_size, 8)
