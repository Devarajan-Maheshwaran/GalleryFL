# GalleryFL 7-Tag Retraining — Summary of Work (2026-07-19)

## 1. HEAD Verification
- Confirmed: 5a966edd4dcbccc514fe59ff70b435f610d5dfd4 ("Android bug fix")

## 2. Cleanup Performed
**Deleted (with justification before removal):**
- All 9 obsolete 34-leaf training scripts (COCO/bootstrap focused):
  - server/scripts/train_head_bootstrap.py
  - server/scripts/retrain_head.py
  - server/scripts/build_coco_features.py
  - server/scripts/build_gallery_tags.py
  - server/scripts/reset_and_retrain.py
  - server/prep_model.py
  - server/prep_eval.py
  - server/coco_to_fgt_mapping.py
  - server/convert_dataset.py

**Reason:** Hard-coded 34 classes, tied to COCO minitrain that produced macro F1 ~0.048. Training-only files.

## 3. New Clean Pipeline (`server/retrain/`)
- `config.py` — **DATASET_DIR** as single source of truth (env + CLI)
- `prepare_dataset.py` — stratified train/val/test manifests
- `train.py` — frozen backbone + 7-class head, **best checkpoint by macro F1**
- `evaluate.py` — full metrics + per-class + optional threshold tuning
- `export_model.py` — final `head_weights.npz` + version bump + reports

## 4. Labels (7 parents)
people, places, activities, objects, documents, nature, events

## 5. Exact User Workflow After Pull
```bash
export DATASET_DIR=/path/to/7tag/dataset
cd server/retrain
python prepare_dataset.py --dataset-dir $DATASET_DIR
python train.py --dataset-dir $DATASET_DIR
python evaluate.py --checkpoint ../output/retrain/best_checkpoint.npz --tune-thresholds
python export_model.py
```

## 6. Final Artifacts
- `server/models/head_weights.npz` (ready for Android + FL)
- `server/output/retrain/retrain_metrics.json` (macro F1 + per-class)
- `server/models/model_version.txt`

## 7. Deliverables Created
- RETRAIN_TRUTH_MAP.md
- RETRAIN_CLEANUP_PLAN.md
- RETRAINING_GUIDE.md
- RETRAIN_AUDIT_NOTES.md
- server/retrain/README.md (detailed)

All other files (Android, server runtime, dashboard, docs, taxonomy.json) untouched.

**Ready state achieved.**
