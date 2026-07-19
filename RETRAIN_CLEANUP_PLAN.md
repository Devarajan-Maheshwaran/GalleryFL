# GalleryFL 7-Tag Retraining — File-by-File Cleanup Plan

**Date:** 2026-07-19  
**HEAD:** 5a966edd4dcbccc514fe59ff70b435f610d5dfd4 (verified)  
**Goal:** Remove ONLY the obsolete 34-leaf training pipeline. Replace with clean 7-parent tag pipeline.  
**Constraints:** 
- Do NOT touch runtime (FL, server, Android, dashboard).
- Keep all *.md files.
- Keep base_model.tflite, model_schema (schema will be regenerated for 7 classes on export if needed).
- Minimal changes outside training.

## 1. Truth Map Summary (see RETRAIN_TRUTH_MAP.md for full)

### DELETE (obsolete 34-tag training pipeline)
All of these are training-only, hardcode 34 classes, rely on COCO 10k mappings, and are the source of the 0.048 F1 disaster.

| File | Justification |
|------|---------------|
| server/scripts/train_head_bootstrap.py | 34-class hardcoded; synthetic + seed loader for old taxonomy |
| server/scripts/retrain_head.py | Duplicate of above; 34-class focal loss retrain |
| server/scripts/build_coco_features.py | COCO→34 leaf mapping + tflite feature extraction for 34 classes |
| server/scripts/build_gallery_tags.py | YOLO version of same 34-leaf feature builder |
| server/scripts/reset_and_retrain.py | 34-class reset + bootstrap logic tied to old prep |
| server/prep_model.py | Old TF bootstrap (MobileNet + 34-class CSV pipeline) |
| server/prep_eval.py | 34-class numpy eval probe logic (used in coordinator but will be replaced by new evaluate) |
| server/coco_to_fgt_mapping.py | Pure COCO→34-leaf dict; obsolete |
| server/convert_dataset.py | COCO minitrain → 34-tag bootstrap CSV converter |

**Deletion rule followed:** Listed + justified before any rm. These files are not referenced by live paths (main.py, model_manager, Android, etc. only use models/*.npz + taxonomy at runtime).

### KEEP (everything else)
- All runtime Python: main.py, fl_coordinator.py, model_manager.py, ws_manager.py, security.py, tag_demand.py, taxonomy_parser.py (lightly extendable)
- config.py / config.json (will only add optional retrain paths)
- taxonomy.json (already perfect — 7 parents)
- models/ (base_model.tflite + schema stay; npz heads will be overwritten by new pipeline)
- scripts/package_server.py, run_*.*
- server/output/ (will be used by new pipeline)
- ALL *.md (including this)
- android/, dashboard/, root docs, tests (non-training)
- requirements.txt (may append light training-only notes)

### REPLACE / ADD
New location: `server/retrain/` (clean isolation)

New files (see implementation below):
- retrain/config.py
- retrain/prepare_dataset.py
- retrain/train.py
- retrain/evaluate.py
- retrain/export_model.py
- retrain/README.md

## 2. Step-by-Step Execution Plan (this run)

1. **Verify HEAD** (done).
2. **Write truth + plan docs** (done).
3. **List files to delete with justification** (this file + truth map).
4. **Perform deletions** (rm only the 9 training files listed).
5. **Create server/retrain/ directory + 5 scripts + config + README**.
6. **Update minimal shared files** (if any):
   - server/config.json: add `retrain_dir`
   - server/taxonomy_parser.py: add optional `get_parent_labels()` (non-breaking)
7. **Update root README** with pointer to new retrain/README.md (never overwrite full docs).
8. **Create final summary docs** (RETRAINING_GUIDE.md).
9. **Test basic script invocation** (no real dataset — smoke test syntax + help).
10. **Leave repo in "pull → set DATASET_DIR → train" state**.

## 3. New 7-Tag Pipeline Design

**Labels (exactly 7 parents):**
```python
LABELS = ["people", "places", "activities", "objects", "documents", "nature", "events"]
NUM_CLASSES = 7
```

**Dataset expectation (user provides):**
```
DATASET_DIR/
  people/
    *.jpg ...
  places/
    ...
  ...
```
(Or a manifest CSV; prepare_dataset supports both.)

**Splits:** 70/15/15 or 80/10/10 (configurable). Stratified by class.

**Backbone:** Use existing `models/base_model.tflite` (frozen 1024-d projection) for bit-compatibility with Android FeatureExtractor. No retraining of backbone.

**Training approach:**
- Extract features once (or on-the-fly).
- Small head: 1024 → 256 (ReLU) → 7 (sigmoid or softmax; we use sigmoid + per-class for multi but enforce top-1 album rule later).
- Loss: BCE with class weights (or focal).
- Optimizer: Adam.
- Best model selected by **macro F1 on val**.
- Per-class threshold tuning on val (optimize F1 per class or global).
- Early stopping on macro F1.

**Outputs:**
- `models/head_weights.npz` (w1, b1, w2, b2) — 7-class version
- `models/model_version.txt` (bumped)
- `output/retrain_metrics.json` (macro_f1, per_class, best_epoch, thresholds, etc.)
- `output/best_checkpoint.npz`

**Compatibility:**
- Same WeightSerializer format used by Android.
- Server model_manager will load it (no change needed if we keep 4-layer npz).
- taxonomy.json parents will be used for album names.

## 4. Explicit Dataset Config Location
Primary (new):
- `server/retrain/config.py` → `DATASET_DIR = os.environ.get("DATASET_DIR", "../data/gallery_7tag")`

Also supported:
- CLI arg `--dataset-dir`
- server/config.json "retrain_dataset_dir"

## 5. Commands After Pull (user only does)
```bash
export DATASET_DIR=/path/to/my/7tag/dataset
cd server/retrain
python prepare_dataset.py --dataset-dir $DATASET_DIR --output manifests/
python train.py --dataset-dir $DATASET_DIR --epochs 30
python evaluate.py --checkpoint output/best_checkpoint.npz
python export_model.py --checkpoint output/best_checkpoint.npz
```

## 6. Logical Errors / Improvements Addressed
- Old: 34 leaves on COCO → poor macro F1 (0.048) because many leaves have almost no examples and are visually ambiguous.
- New: 7 coarse parents → much higher separability + macro F1.
- Personalization: taxonomy.json already supports children. Android will use sub-tags from user feedback/organize under the 7 parents.
- One-image-one-folder: already partially in OrganizeExecutor; new training will output confidence scores so ranking can pick the single best parent album.
- No fake metrics: new evaluate always computes real macro-F1, per-class, support.
- Duplicate training scripts removed.
- No notebook-only; everything is plain .py with argparse.
- Dataset path single source of truth.

## 7. Post-Cleanup State
After this:
- `server/retrain/` contains the **only** training entry point.
- Old scripts gone.
- User runs only the 4 scripts in retrain/.
- Artifacts go to the same `models/` and `output/` so existing FL/Android paths continue to work without modification.

**Ready for user:** set DATASET_DIR, run pipeline, get 7-tag head.

END PLAN