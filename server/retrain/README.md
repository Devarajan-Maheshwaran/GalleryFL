# GalleryFL 7-Parent Retraining (Clean Pipeline)

## 1. Dataset Preparation (for Kaggle coco-minitrain-10k)

The Kaggle dataset has this exact structure:

```
coco_minitrain_10k/
├── images/
│   ├── train2017/
│   └── val2017/
├── labels/
│   ├── train2017/
│   └── val2017/
├── train2017.txt
└── val2017.txt
```

**Convert it to the 7-parent format** (required by the trainer):

```bash
# From the GalleryFL root
cd server/retrain

python convert_coco.py \
    --coco_root /path/to/coco_minitrain_10k \
    --output ~/data/gallery_7tag \
    --max-per-class 300 \
    --splits train2017,val2017
```

**Recommended for first training run:**
- Use `--max-per-class 250` or `300` (much faster)
- Use `--use-symlinks` if you are on the same filesystem (very fast, no copying)

After conversion you will have:

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

## 2. Train the 7-Parent Model

```bash
export DATASET_DIR=~/data/gallery_7tag

cd server/retrain

# 1. Create proper train/val splits
python prepare_dataset.py --dataset-dir $DATASET_DIR

# 2. Train (selects best model by macro F1)
python train.py --dataset-dir $DATASET_DIR --epochs 20

# 3. Evaluate with per-class metrics
python evaluate.py \
    --checkpoint ../output/retrain/best_checkpoint.npz \
    --tune-thresholds

# 4. Export final production model (head_weights.npz)
python export_model.py
```

## 3. What You Get

- `server/models/head_weights.npz` ← **The model you use**
- `server/output/retrain/retrain_metrics.json`
- `server/models/model_version.txt`

## Full One-Liner Example

```bash
export DATASET_DIR=~/data/gallery_7tag

cd GalleryFL/server/retrain

python convert_coco.py \
    --coco_root ~/data/coco_minitrain_10k \
    --output $DATASET_DIR \
    --max-per-class 250

python prepare_dataset.py --dataset-dir $DATASET_DIR
python train.py --dataset-dir $DATASET_DIR --epochs 15
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds
python export_model.py
```

## Requirements

```bash
pip install numpy tensorflow-cpu pillow
```

## Notes

- The converter uses majority-vote on YOLO labels to assign each image to one of the 7 parents.
- `documents` and `events` will be very small from COCO (this is expected).
- Real quality for those classes will come later from on-device FL + user photos.
