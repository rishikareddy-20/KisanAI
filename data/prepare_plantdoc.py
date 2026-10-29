"""Build a PlantDoc field test set aligned to PlantVillage labels.

PlantDoc has its own class names and train/test split; we remap its folders to
PlantVillage names via data/plantdoc_to_plantvillage.json and write them all into
one test/ split. By default both PlantDoc splits are combined (we never train on
PlantDoc here, so there's no leakage).

    python -m data.prepare_plantdoc
    python -m data.prepare_plantdoc --src "C:/path/to/PlantDoc-Dataset-..." --splits train test
    python -m data.prepare_plantdoc --dst data/processed_plantdoc

    python -m training.evaluate --data data/processed_plantdoc   # then evaluate
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_SRC = Path.home() / "Downloads" / "PlantDoc-Dataset-windows-compatible-master"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default=str(DEFAULT_SRC), help="PlantDoc dataset root")
    parser.add_argument("--dst", default="data/processed_plantdoc", help="output dir")
    parser.add_argument(
        "--splits", nargs="+", default=["train", "test"],
        help="which PlantDoc splits to include in the field test set",
    )
    parser.add_argument(
        "--map", default="data/plantdoc_to_plantvillage.json", help="mapping file",
    )
    args = parser.parse_args()

    src = Path(args.src)
    if not src.is_absolute():
        src = ROOT_DIR / src
    if not src.exists():
        raise SystemExit(f"PlantDoc source not found: {src}")

    mapping = json.loads((ROOT_DIR / args.map).read_text(encoding="utf-8"))["map"]
    out_test = (ROOT_DIR / args.dst) / "test"

    copied = 0
    skipped_classes = []
    per_class: dict[str, int] = {}

    for split in args.splits:
        split_dir = src / split
        if not split_dir.exists():
            print(f"(skipping missing split: {split_dir})")
            continue
        for class_dir in sorted(d for d in split_dir.iterdir() if d.is_dir()):
            pv_key = mapping.get(class_dir.name, "__MISSING__")
            if pv_key == "__MISSING__":
                skipped_classes.append(class_dir.name)
                continue
            if pv_key is None:  # explicitly excluded (e.g. "duplicates")
                continue
            out_dir = out_test / pv_key
            out_dir.mkdir(parents=True, exist_ok=True)
            for img in class_dir.iterdir():
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                # Prefix with split to avoid name collisions between train/test.
                target = out_dir / f"{split}__{img.name}"
                if not target.exists():
                    shutil.copy2(img, target)
                copied += 1
                per_class[pv_key] = per_class.get(pv_key, 0) + 1

    print(f"\n✅ PlantDoc field test set written to {out_test}")
    print(f"   {copied} images across {len(per_class)} PlantVillage-aligned classes")
    for k in sorted(per_class):
        print(f"     {k}: {per_class[k]}")
    if skipped_classes:
        print(f"\n⚠️  Unmapped PlantDoc folders (skipped): {sorted(set(skipped_classes))}")
        print("   Add them to data/plantdoc_to_plantvillage.json to include them.")


if __name__ == "__main__":
    main()
