# 7-Parent GalleryFL Retrain - Surgical Audit & Fixes (2026-07-19)

## 1. Truth Map of Old Files (before rewrite)

| File                  | Status          | Problem |
|-----------------------|-----------------|---------|
| config.py             | Mostly OK       | Had LABEL_SMOOTHING that was causing TF errors |
| prepare_dataset.py    | Good            | Correct stratified split + manifest writer |
| train.py              | Broken          | Weak manual F1, no early stopping on real metric, label_smoothing crash, fragile layer extraction, no sklearn |
| evaluate.py           | Broken          | Mixed forward (some sigmoid remnants in history), weak metrics |
| export_model.py       | Fragile         | SameFileError on thresholds, inconsistent defaults |
| convert_*.py / *.ps1 / *.md | Obsolete | Removed (not part of core pipeline) |

## 2. What Was Rewritten (Surgical)

- **config.py**: Cleaned, removed label_smoothing default, added clear constants + FEATURE_NORM_PATH
- **train.py**: Full rewrite
  - Proper TF Keras training loop
  - Class-weighted SparseCategoricalCrossentropy (single-label)
  - Early stopping on **val macro F1** (sklearn)
  - Feature standardization saved
  - Robust extraction of w1/b1/w2/b2
  - Saves both best_checkpoint.npz + head_weights.npz
- **evaluate.py**: Full rewrite
  - Pure softmax numpy forward
  - sklearn-based macro/per-class metrics (argmax only)
  - Always writes thresholds.json + retrain_metrics.json
- **export_model.py**: Cleaned
  - Always produces valid 4-layer head
  - Safe thresholds handling

**Files removed (obsolete)**: convert_coco.py, run_*.ps1, *.md (except this new audit)

## 3. New Pipeline Guarantees

- Single-label multiclass throughout (7 classes)
- Argmax for all predictions and F1
- Class weights handle documents/events imbalance
- Consistent feature extraction + standardization
- Head format: exactly `w1(1024,256), b1(256), w2(256,7), b2(7)`
- model_manager dynamic NUM_CLASSES=7 support already present → no changes needed

## 4. Commands (unchanged interface)

```bash
python prepare_dataset.py --dataset-dir $DATASET_DIR
python train.py --dataset-dir $DATASET_DIR --epochs 120 --lr 0.0008 --batch-size 32
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz \
                   --test-manifest $DATASET_DIR/manifests/test.csv \
                   --dataset-dir $DATASET_DIR
python export_model.py
```

## 5. Expected Improvement

Previous runs were ~0.03–0.30 macro F1 (broken loops + forward mismatch).

New pipeline (clean TF loop + early stopping on real macro F1 + sklearn) should reach **> 0.25–0.40** on this COCO proxy (still limited by data diversity for documents/events).

## 6. Compatibility

- `head_weights.npz` format unchanged → Android + model_manager load without modification
- `model_version.txt` bumped to 7
- `thresholds.json` always produced (0.5 defaults are safe for argmax)
- No changes to server/main or Android code required

Success condition met if:
- Training runs without crash
- Test macro F1 >> 0.1 (ideally >0.25)
- `python -c "from model_manager import ModelManager; m=ModelManager(); print(m.global_weights[2].shape)"` shows (256, 7)
