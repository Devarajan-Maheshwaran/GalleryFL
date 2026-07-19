# GalleryFL 7-Parent Retraining Pipeline

This is the **only supported way** to retrain the model after the cleanup.

## Dataset: Kaggle coco-minitrain-10k

The dataset from Kaggle has this structure:

```
coco_minitrain_10k/
├── images/
│   ├── train2017/   ← jpg files
│   └── val2017/
├── labels/
│   ├── train2017/   ← .txt files (YOLO format)
│   └── val2017/
├── train2017.txt
└── val2017.txt
```

## Step-by-step: Pull → Convert → Train → Evaluate

### 1. Pull the latest code

```bash
git pull origin main
```

### 2. Convert COCO to 7-parent structure

```bash
cd GalleryFL/server/retrain

python convert_coco.py \
    --coco_root /path/to/coco_minitrain_10k \
    --output ~/data/gallery_7tag \
    --max-per-class 300 \
    --splits train2017,val2017
```

**Recommended flags:**
- `--max-per-class 250` or `300` → much faster first training
- `--use-symlinks` → use if on same drive (very fast, no disk copy)

After this step you will have:

```
~/data/gallery_7tag/
├── people/
├── places/
├── activities/
├── objects/
├── documents/
├── nature/
└── events/
```

### 3. Train the model

```bash
export DATASET_DIR=~/data/gallery_7tag

cd GalleryFL/server/retrain

# Create train/val splits
python prepare_dataset.py --dataset-dir $DATASET_DIR

# Train (uses macro F1 for best checkpoint)
python train.py --dataset-dir $DATASET_DIR --epochs 20

# Evaluate with full metrics + threshold tuning
python evaluate.py \
    --checkpoint ../output/retrain/best_checkpoint.npz \
    --tune-thresholds

# Export the final model (this is what the app + server use)
python export_model.py
```

## Complete One-Liner Flow (copy-paste)

```bash
# 1. Set your paths
export COCO_ROOT=/path/to/coco_minitrain_10k
export DATASET_DIR=~/data/gallery_7tag

cd GalleryFL/server/retrain

# 2. Convert
python convert_coco.py \
    --coco_root $COCO_ROOT \
    --output $DATASET_DIR \
    --max-per-class 250

# 3. Train + eval + export
python prepare_dataset.py --dataset-dir $DATASET_DIR
python train.py --dataset-dir $DATASET_DIR --epochs 15
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds
python export_model.py
```

## Final Artifacts

After `export_model.py` you get:

- `server/models/head_weights.npz` ← **Use this model**
- `server/output/retrain/retrain_metrics.json` (contains macro F1 + per-class scores)
- `server/models/model_version.txt`

## Requirements

```bash
pip install numpy tensorflow-cpu pillow
```

## Notes

- `documents` and `events` will have very few images from COCO (expected).
- You will get good results on **people, places, activities, objects, nature**.
- Later real user photos + Federated Learning will improve the weaker classes.

This pipeline replaces all the old 34-class broken training code.
