# GalleryFL seven-parent training pipeline

This is the only centralized training entry point. It trains a **single-label,
seven-class** head for:

`people, places, activities, objects, documents, nature, events`

The existing two-output TFLite backbone remains frozen. The learned head is
`1024 -> 256 ReLU -> 7 logits`. Training uses class-weighted sparse categorical
cross-entropy. Validation and test predictions always use `argmax`.

## Dataset

```text
$DATASET_DIR/
  people/
  places/
  activities/
  objects/
  documents/
  nature/
  events/
```

Images may be nested below each class folder. Every class needs at least ten
unique images. Exact duplicates are removed before splitting; conflicting
labels for identical image content are rejected.

The Kaggle **COCO Minitrain 10K** YOLO layout is also auto-detected:

```text
$DATASET_DIR/
  images/{train2017,val2017}/
  labels/{train2017,val2017}/
```

Because COCO has 80 object labels rather than GalleryFL parent labels,
`prepare_dataset.py` applies a versioned, exhaustive proxy map recorded in
`manifests/split_summary.json`. It uses 85%/15% of COCO train2017 for train/val
and keeps COCO val2017 untouched as test. This proxy is suitable for global-head
initialization and benchmarking, but native gallery parent labels remain the
preferred production dataset.

## Commands

Run these commands from `server/retrain`:

```bash
python prepare_dataset.py --dataset-dir "$DATASET_DIR"
python train.py --dataset-dir "$DATASET_DIR" --epochs 100 --lr 0.0008 --batch-size 32
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --test-manifest "$DATASET_DIR/manifests/test.csv" --dataset-dir "$DATASET_DIR"
python export_model.py
```

The evaluation command writes the full accuracy, macro F1, per-class
precision/recall/F1, confusion matrix, and a raw-feature test cache used by the
FL server. It exits with code 2 if test macro F1 is below 0.20, but still writes
the report for diagnosis. Do not export a checkpoint that fails this gate.

## PowerShell

```powershell
$env:DATASET_DIR = "C:\path\to\gallery_7parent"
Set-Location .\server\retrain
python .\prepare_dataset.py --dataset-dir $env:DATASET_DIR
python .\train.py --dataset-dir $env:DATASET_DIR --epochs 100 --lr 0.0008 --batch-size 32
python .\evaluate.py --checkpoint ..\output\retrain\best_checkpoint.npz --test-manifest "$env:DATASET_DIR\manifests\test.csv" --dataset-dir $env:DATASET_DIR
if ($LASTEXITCODE -ne 0) { throw "Evaluation quality gate failed; inspect ..\output\retrain\test_metrics.json" }
python .\export_model.py
```

## Outputs

Training and evaluation outputs:

- `server/output/retrain/best_checkpoint.npz` — normalized-space training head
  plus train-only feature statistics.
- `server/output/retrain/train_metrics.json` — validation history and class
  weights.
- `server/output/retrain/test_metrics.json` — held-out test report.
- `server/output/retrain/test_features.npz` — raw frozen-backbone test features
  for framework-free FL round evaluation.
- `server/output/retrain/thresholds.json` — argmax decision metadata. Thresholds
  are optional post-argmax abstention gates, never multi-label decisions.

Deployment outputs from `export_model.py`:

- `server/models/head_weights.npz` — exactly `w1`, `b1`, `w2`, `b2`.
- `server/models/model_schema.json` — seven-class runtime contract.
- `server/models/model_version.txt` — incremented deployment version.
- `server/models/thresholds.json` — matching single-label decision metadata.

## Feature consistency

The trainer standardizes frozen backbone features with statistics computed from
the training split only. Export algebraically folds that transform into `w1`
and `b1`. Therefore Android and the FL server continue passing raw 1024-D
features to the unchanged four-tensor head format—no extra normalization tensor
or runtime sidecar is required.

`server/models/base_model.tflite` and the Android asset are byte-identical. Both
training and Android select the explicit `[1, 1024]` projection output; the
`[1, 7, 7, 1024]` output remains available for heatmaps.
