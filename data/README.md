# Data

This folder holds the datasets and the preprocessing pipeline. Raw and processed
images are git-ignored because they're large, so you download and generate them
locally.

## Datasets

### 1. PlantVillage (base training, clean lab images)
- ~54,000 images across 38 classes.
- Common sources: [Kaggle PlantVillage Dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset), or the "New Plant Diseases Dataset".

The Kaggle download unzips to three renderings of the same leaves:
`plantvillage dataset/{color, grayscale, segmented}/`, each with the 38 class
folders. Use `color` only. It's natural RGB, which matches the inference pipeline.
`grayscale` drops colour, which is a key disease signal, and `segmented` blacks out
the background, unlike real phone photos. Merging all three triples the data with
near-duplicates and causes train/test leakage.

Copy the 38 folders from `color/` directly into `data/raw/` (no wrapper folder):

```
data/raw/
  Apple___Apple_scab/
  Apple___Black_rot/
  Tomato___Late_blight/
  ...   (38 folders, from color/)
```

On Windows, e.g.:

```bat
robocopy "%USERPROFILE%\Downloads\archive\plantvillage dataset\color" "data\raw" /E /NFL /NDL
```

> Label naming: PlantVillage folder names (for example `Corn_(maize)___Common_rust_`,
> `Tomato___Tomato_Yellow_Leaf_Curl_Virus`) differ from the canonical KB keys. The
> model trains on the original folder names; at advisory time the retriever resolves
> them via [`knowledge_base/label_map.json`](../knowledge_base/label_map.json). 17 of
> the 38 classes map to curated KB entries today; the other 21 (mostly other crops)
> fall back to the "consult an expert" message until the KB is extended.

### 2. PlantDoc (field validation, noisy real-world images)
- ~2,500 real field images with occlusion and noise.
- Source: [PlantDoc dataset (GitHub)](https://github.com/pratikkayal/PlantDoc-Dataset).
- The download has its own `train/` and `test/` folders, and its class names differ
  completely from PlantVillage (for example `Tomato Early blight leaf`).

We use PlantDoc only as a field test set (we do not train on it), so a model trained
on PlantVillage can be scored on messy real-world photos. A prep
script remaps PlantDoc's folders to the PlantVillage label names via
[`plantdoc_to_plantvillage.json`](plantdoc_to_plantvillage.json) and writes them
into a `test/` split:

```bash
python -m data.prepare_plantdoc --src "C:/Users/<you>/Downloads/PlantDoc-Dataset-windows-compatible-master"
# → data/processed_plantdoc/test/<PlantVillage_class>/*.jpg
```

(The `--src` default already points at `~/Downloads/PlantDoc-Dataset-windows-compatible-master`.)

Then measure field robustness against the same trained model:

```bash
python -m training.evaluate --data data/processed_plantdoc
```

`evaluate.py` reconciles PlantDoc's class subset with the model's label space by
name and reports macro/per-class F1 over the classes present. This is your field-F1,
the real-world number to compare against the clean lab-F1.

## Preprocess into train/val/test

```bash
# Full PlantVillage split
python -m data.prepare --src data/raw --dst data/processed

# Quick sample run (20 images/class) to validate the pipeline fast
python -m data.prepare --src data/raw --dst data/processed --sample 20 --copy

# PlantDoc as a separate field test set
python -m data.prepare --src data/raw_plantdoc --dst data/processed_plantdoc --copy
```

This produces:

```
data/processed/
  train/<class>/*.jpg
  val/<class>/*.jpg
  test/<class>/*.jpg
```

which `training/train.py` and `training/evaluate.py` consume directly.

## Recommended workflow
1. `data.prepare --sample 20` → `training.train --sample` → confirm the full
   path works end-to-end (data → train → eval → export).
2. Then run the full `data.prepare` and `training.train` on a GPU (Colab is fine).
3. Evaluate on both `data/processed/test` (clean) and `data/processed_plantdoc/test`
   (field) to report lab-F1 vs. field-F1.
