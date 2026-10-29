"""Split raw per-class folders (data/raw/) into a train/val/test ImageFolder
layout (data/processed/). Split is stratified per class.

    python -m data.prepare
    python -m data.prepare --sample 20        # tiny subset for the sample run
    python -m data.prepare --src data/raw_plantdoc --dst data/processed_plantdoc
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path
from typing import List

ROOT_DIR = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_images(class_dir: Path) -> List[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in IMG_EXTS]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="data/raw", help="raw dataset dir (per-class folders)")
    parser.add_argument("--dst", default="data/processed", help="output dir")
    parser.add_argument("--train", type=float, default=0.7)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.15)
    parser.add_argument("--sample", type=int, default=0, help="max images/class (0 = all)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="copy instead of symlink")
    args = parser.parse_args()

    assert abs(args.train + args.val + args.test - 1.0) < 1e-6, "splits must sum to 1.0"

    random.seed(args.seed)
    src = (ROOT_DIR / args.src).resolve()
    dst = (ROOT_DIR / args.dst).resolve()
    if not src.exists():
        raise SystemExit(f"Source not found: {src}\nSee data/README.md for how to download datasets.")

    class_dirs = sorted([d for d in src.iterdir() if d.is_dir()])
    if not class_dirs:
        raise SystemExit(f"No class folders in {src}")

    totals = {"train": 0, "val": 0, "test": 0}
    for class_dir in class_dirs:
        images = list_images(class_dir)
        random.shuffle(images)
        if args.sample:
            images = images[: args.sample]
        n = len(images)
        n_train = int(n * args.train)
        n_val = int(n * args.val)
        splits = {
            "train": images[:n_train],
            "val": images[n_train : n_train + n_val],
            "test": images[n_train + n_val :],
        }
        for split, files in splits.items():
            out_dir = dst / split / class_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in files:
                target = out_dir / f.name
                if target.exists():
                    continue
                if args.copy:
                    shutil.copy2(f, target)
                else:
                    try:
                        target.symlink_to(f)
                    except OSError:
                        shutil.copy2(f, target)  # Windows without symlink privilege
            totals[split] += len(files)

    print(f"Prepared {len(class_dirs)} classes into {dst}")
    print(f"  train={totals['train']}  val={totals['val']}  test={totals['test']}")


if __name__ == "__main__":
    main()
