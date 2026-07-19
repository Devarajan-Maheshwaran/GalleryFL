# GalleryFL Retraining Truth Map (HEAD 5a966ed verified 2026-07-19)

**Repo HEAD verified:** 5a966edd4dcbccc514fe59ff70b435f610d5dfd4 ("Android bug fix")

## Scope of this retrain task
- Replace **only** the obsolete 34-leaf training pipeline.
- Build a clean, production-style **7-parent tag** classifier (people, places, activities, objects, documents, nature, events).
- Goal: dramatically better macro-F1 than the 0.048 from COCO 10k + 34 leaves.
- Keep everything else (FL runtime, server, Android, dashboard, .md docs, base_model.tflite, etc.).
- Output must be drop-in compatible with existing head format (w1/b1/w2/b2) so Android + server can consume a 7-class head.

## Training-related files — current state (HEAD)

### OBSOLETE / 34-LEAF TRAINING PIPELINE (TO BE REMOVED)
These are tied exclusively to the 34-leaf taxonomy + COCO bootstrap flow that produced the bad F1=0.048. They hardcode 34 classes, use old synthetic/COCO mappings, and duplicate logic.

- `server/scripts/train_head_bootstrap.py` — 34-class hardcoded, focal loss synthetic/seed loader
- `server/scripts/retrain_head.py` — identical 34-class synthetic retrain
- `server/scripts/build_coco_features.py` — full COCO 34-leaf mapping + feature extraction
- `server/scripts/build_gallery_tags.py` — YOLO 34-leaf mapping + feature extraction
- `server/scripts/reset_and_retrain.py` — 34-class reset + calls into old prep
- `server/prep_model.py` — old TF full-model bootstrap using 34-leaf CSVs + taxonomy
- `server/prep_eval.py` — 34-class numpy eval (used by coordinator)
- `server/coco_to_fgt_mapping.py` — COCO→34 FGT leaf map
- `server/convert_dataset.py` — COCO→34 bootstrap CSV converter

**Justification for deletion:** All hardcode `NUM_CLASSES=34` or rely on `TaxonomyParser().num_classes==34`, contain COCO-specific leaf mappings, and are the direct cause of the fragile low-F1 34-tag setup. They are training-only and not used by live FL/runtime paths.

### KEEP (RUNTIME / SHARED / NON-TRAINING)
- `server/main.py`, `fl_coordinator.py`, `model_manager.py`, `ws_manager.py`, `security.py`, `tag_demand.py`
- `server/config.py`, `config.json` (minimal updates only for output paths)
- `server/taxonomy.json` (already uses 7 parents — perfect for new UI + personalization)
- `server/taxonomy_parser.py` (keep; may be lightly extended later for parent mapping)
- `server/models/base_model.tflite`, `model_schema.json` (backbone + schema stay)
- `server/requirements.txt` (add nothing heavy)
- All `server/dashboard/`, `android/`, root + server `*.md`, tests (unless training-only)
- `server/output/` (will be cleaned by new pipeline)
- `server/scripts/package_server.py`, `run_server.*` (deployment, not training)

### REPLACE / NEW (7-TAG CLEAN PIPELINE)
New focused training code will live under:
- `server/retrain/` (clean separation — no pollution of server root)

Contents (new):
- `server/retrain/config.py` — single source `DATASET_DIR`, splits, 7 labels
- `server/retrain/prepare_dataset.py` — folder-per-class → train/val/test splits + manifest
- `server/retrain/train.py` — feature extraction (frozen backbone) + 7-class head training (macro-F1 best checkpoint, class weights, threshold tuning)
- `server/retrain/evaluate.py` — full metrics (macro F1, per-class F1/prec/rec, confusion summary)
- `server/retrain/export_model.py` — produce `models/head_weights.npz` (7-class) + metrics report + optional tflite head
- `server/retrain/README.md` — exact commands (this becomes the retraining doc)

### TAXONOMY STRATEGY
- `taxonomy.json` already declares the exact 7 parents. New pipeline trains **on the 7 parent IDs** (single-label or soft multi but argmax for album assignment).
- Leaves remain for future personalization / sub-tagging in the Android app (Non-IID user data + feedback will populate sub-albums under e.g. Nature/Beach).
- One-image-one-folder rule will be enforced at album creation time using model ranking + confidence (existing OrganizeExecutor logic + new top-1).

### CURRENT BROKEN STATE (why retrain)
- Old pipeline produced macro F1 ~0.048 on 34 leaves from COCO minitrain (class imbalance + fine-grained leaves the model couldn't distinguish reliably).
- Synthetic data in scripts was never realistic.
- No explicit DATASET_DIR, no proper splits in many paths, no macro-F1 driven checkpointing.
- No per-class threshold tuning or balancing.

### SUCCESS TARGETS FOR NEW PIPELINE
- Train on real user/gallery images organized in 7 folders.
- Macro F1 primary metric (target >> 0.7 on reasonable data).
- Best checkpoint by macro F1 on val.
- Output:
  - `models/head_weights.npz` (w1 1024×256, b1, w2 256×7, b2)
  - `output/retrain_metrics.json` (full report)
  - `output/best_checkpoint.npz`
- Easy to swap back into existing FL flow (same serialization, same head shape contract).

## Files that will be touched (minimal)
- Delete the 9 obsolete training files listed above.
- Create `server/retrain/` + 5 core scripts + config + README.
- Minor updates:
  - `server/config.json` (add `retrain_output_dir`)
  - `server/taxonomy_parser.py` (optional parent support, non-breaking)
  - Possibly `server/model_manager.py` for 7-class awareness (if needed — keep minimal)
- All .md files untouched except new RETRAIN_* docs.
- Android/server runtime **untouched** except the model artifacts they will load.

## Verification performed before edits
- HEAD: 5a966ed confirmed.
- No other uncommitted training changes.
- All obsolete scripts inspected (hard 34-class assumptions confirmed).
- taxonomy.json already 7-parent friendly.

**End of Truth Map**