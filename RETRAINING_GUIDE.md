# GalleryFL — 7-Tag Retraining Guide (Clean Pipeline)

**Repo state after cleanup:** HEAD 5a966ed + new `server/retrain/` only training entrypoint.

## Quick Start (What You Do After Pulling)

```bash
# ONLY CONFIGURATION NEEDED
export DATASET_DIR=/absolute/path/to/your/7-category/gallery

cd server/retrain

# 1. Prepare splits (recommended)
python prepare_dataset.py --dataset-dir $DATASET_DIR

# 2. Train (macro-F1 driven)
python train.py --dataset-dir $DATASET_DIR --epochs 30

# 3. Evaluate + tune
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds

# 4. Export production artifacts
python export_model.py
```

That's it. You now have a 7-class head ready for the app and FL.

## Where to Set Dataset Directory (SINGLE SOURCE)

**File:** `server/retrain/config.py`

```python
DATASET_DIR = os.environ.get("DATASET_DIR", "...default...")
```

Also supported via:
- `--dataset-dir` CLI flag on every script
- Environment variable `DATASET_DIR`

## Final Output Paths

After `export_model.py`:

- `server/models/head_weights.npz` ← **the one the app + server use**
- `server/models/model_version.txt`
- `server/output/retrain/best_checkpoint.npz`
- `server/output/retrain/retrain_metrics.json`
- `server/output/retrain/final_model_info.json`

## Full Pipeline Scripts

| Script                    | Purpose                              | Key Flags                     |
|---------------------------|--------------------------------------|-------------------------------|
| `prepare_dataset.py`      | Create train/val/test manifests      | `--dataset-dir`, `--seed`     |
| `train.py`                | Feature extraction + head training   | `--epochs`, `--lr`            |
| `evaluate.py`             | Metrics + optional threshold tuning  | `--checkpoint`, `--tune-thresholds` |
| `export_model.py`         | Finalize for deployment              | `--checkpoint`                |

## Dataset Format

```
$DATASET_DIR/
  people/       (jpg/png)
  places/
  activities/
  objects/
  documents/
  nature/
  events/
```

## Metrics Focus

- Primary: **macro F1** (used for best checkpoint)
- Per-class F1 / precision / recall / support
- Top confusions

Old 34-leaf COCO pipeline gave ~0.048 macro F1. New 7-parent approach targets 0.75+ on real data.

## Integration

The exported head is 100% compatible with:
- Android `ClassificationHead` + `WeightSerializer`
- Server FL coordinator
- Existing `taxonomy.json` (7 parents + children for sub-albums)

## Notes on Personalization & One-Image-One-Folder

- `taxonomy.json` already defines children under the 7 parents.
- Android OrganizeExecutor + ranking logic will place each photo into **exactly one** parent album using the highest-confidence prediction.
- User feedback / Non-IID data can later create sub-albums (e.g. nature/beach) without changing the core model.

## Cleanup Performed

All old 34-leaf scripts were **listed + justified** then deleted:
- train_head_bootstrap.py, retrain_head.py, build_*.py, reset_and_retrain.py, prep_model.py, prep_eval.py, coco_to_fgt_mapping.py, convert_dataset.py

Only the new `server/retrain/` pipeline remains.

**You are now ready to retrain.**