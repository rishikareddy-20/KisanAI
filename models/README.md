# Models

Trained artifacts land here (they are git-ignored). The backend loads them
automatically if present, and falls back to a mock classifier if not, so the app
runs end-to-end before you've trained anything.

Expected files (produced by the training pipeline):

| File | Produced by | Used by |
|------|-------------|---------|
| `kisanai_efficientnet.keras` | `training/train.py` | evaluation + export |
| `labels.json` | `training/train.py` | backend (class names) |
| `eval_report.json` | `training/evaluate.py` | metrics record (macro/per-class F1) |
| `confusion_matrix.png` | `training/evaluate.py` | eval report |
| `kisanai_efficientnet.tflite` | `training/export_tflite.py` | backend inference |
| `tts_cache/` | backend (gTTS) | served audio |

To produce them:

```bash
python -m data.prepare --src data/raw --dst data/processed
python -m training.train
python -m training.evaluate
python -m training.export_tflite
```
