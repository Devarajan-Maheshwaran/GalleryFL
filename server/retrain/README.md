# GalleryFL 7-Parent Tag Retraining Pipeline

**Clean, production-ready retraining for the core 7-tag gallery classifier.**

Replaces the old fragile 34-leaf COCO-based training (which produced macro F1 ≈ 0.048).

## Target Taxonomy (7 Parents)

The model is trained on these coarse but highly separable categories:

- **people**
- **places**
- **activities**
- **objects**
- **documents**
- **nature**
- **events**

These map directly to the existing `server/taxonomy.json` parent categories. Fine-grained leaves (e.g. "beach", "mountain" under places) remain for future personalization / Non-IID user feedback.

## Why 7 Parents Instead of 34 Leaves?

- COCO 10k + 34 leaves → very low macro F1 (0.048) due to extreme class imbalance and visual ambiguity of fine tags.
- 7 parents are visually distinct → dramatically higher macro F1.
- One-image-one-folder rule is enforced at album creation time using model ranking + confidence.
- Personalization: user data + feedback can populate sub-albums (e.g. Nature → Beach / Forest) without retraining the core head.

## Prerequisites

1. A dataset organized as:
   ```
   /path/to/your/dataset/
       people/      (hundreds of images)
       places/
       activities/
       objects/
       documents/
       nature/
       events/
   ```

2. `server/models/base_model.tflite` must exist (the frozen MobileNetV3 projection).

3. Python environment with:
   ```bash
   pip install numpy tensorflow-cpu pillow
   ```

## Exact Commands (After Setting Dataset)

```bash
# 1. Set your dataset (only thing you need to configure)
export DATASET_DIR=/path/to/your/7tag/dataset

# 2. (Optional but recommended) Prepare clean splits
cd server/retrain
python prepare_dataset.py --dataset-dir $DATASET_DIR

# 3. Train (selects best checkpoint by macro F1 on val)
python train.py --dataset-dir $DATASET_DIR --epochs 30

# 4. Evaluate
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds

# 5. Export final production artifacts
python export_model.py --checkpoint ../output/retrain/best_checkpoint.npz
```

## Dataset Config Location (SINGLE SOURCE OF TRUTH)

**Primary:**
- `server/retrain/config.py` line ~15:
  ```python
  DATASET_DIR = os.environ.get("DATASET_DIR", "...")
  ```

**Also accepted:**
- CLI flag: `--dataset-dir /path`
- Environment variable `DATASET_DIR`
- Can be extended to read from `server/config.json`

## Final Output Artifacts

After successful run you get:

| Path | Purpose |
|------|---------|
| `server/models/head_weights.npz` | **Production head** (w1/b1/w2/b2) — used by Android + server |
| `server/models/model_version.txt` | Version bump (currently "7") |
| `server/output/retrain/best_checkpoint.npz` | Best training checkpoint (by macro F1) |
| `server/output/retrain/retrain_metrics.json` | Full metrics (macro F1, per-class, etc.) |
| `server/output/retrain/final_model_info.json` | Summary for deployment |

## Metrics Reported

- **macro_f1** (primary selection criterion)
- per-class: f1, precision, recall, support
- accuracy
- top confusions

Example expected output on a decent dataset:
```
Macro F1: 0.82
Per-class F1s all > 0.70
```

## How It Integrates Back Into the Project

- The exported `head_weights.npz` uses the **exact same 4-layer format** as before.
- Android `WeightSerializer` + `ClassificationHead` work unchanged.
- Server `model_manager.py` loads it automatically.
- `taxonomy.json` already defines the 7 parents.
- FL training continues to work (the head is just 7-class now).

## Tips for Best Results

- Minimum ~200-300 images per class for good macro F1.
- Use real user gallery photos (not synthetic).
- Run `prepare_dataset.py` for proper stratified splits.
- Use `--tune-thresholds` in evaluate for better per-class performance.
- After export, restart the server or re-register clients to pick up the new head.

## Cleanup Note

All old 34-leaf training scripts (`prep_model.py`, `train_head_bootstrap.py`, `build_*_features.py`, etc.) have been removed. Only this clean `retrain/` pipeline remains.

## Next Steps After Training

1. Copy `models/head_weights.npz` + `model_version.txt` to your production server.
2. (Optional) Run full FL rounds on real devices for further personalization.
3. Android will automatically use the new 7-class model for "For You" / Explore albums.

**This is now the only supported way to retrain the GalleryFL classifier.**