# Retraining Audit Notes (during 7-tag pipeline preparation)

**Performed at HEAD 5a966edd... on 2026-07-19**

## Issues Discovered & Addressed in Training Pipeline

1. **Old training was 34-leaf only**
   - Every training script hard-coded or derived `NUM_CLASSES=34` from taxonomy.
   - Caused the reported macro F1 0.048 on COCO minitrain (too many fine-grained, poorly covered leaves).
   - **Fixed by deletion + new 7-class pipeline.**

2. **No single DATASET_DIR**
   - Scattered defaults: `../data/coco_minitrain_10k`, `../data/bootstrap_seed`, hard-coded in many places.
   - **Fixed:** `server/retrain/config.py` is now the single source + env + CLI.

3. **Duplicate training implementations**
   - `train_head_bootstrap.py`, `retrain_head.py`, `prep_model.py` all did almost identical synthetic/focal head training.
   - Removed.

4. **No macro-F1 driven checkpointing**
   - Old code mostly used loss or accuracy.
   - New `train.py` selects best by macro F1 on validation.

5. **Missing proper split tooling**
   - Many paths assumed pre-made CSVs without clear generation.
   - `prepare_dataset.py` now creates stratified manifests.

6. **Evaluation was probe-based synthetic or incomplete**
   - `prep_eval.py` used fixed synthetic probe.
   - New `evaluate.py` works on real held-out data + reports full per-class metrics.

7. **COCO-specific dead code**
   - `coco_to_fgt_mapping.py` + `convert_dataset.py` + feature builders were 100% tied to 34-leaf COCO flow.
   - Deleted.

## One-Image-One-Folder + Personalization Strategy (Verified)

- `server/taxonomy.json` already correctly structures 7 parents + children.
- Android side (from prior work):
  - `ExploreScreen` + `ClassificationHead.forward` + `OrganizeExecutor` already select **top class** per image.
  - One photo ends up in one primary album (ranking by margin/confidence).
- Future sub-albums (beaches under nature) are populated via:
  - User feedback
  - `LocalFeedbackStore`
  - Tag signal / demand
- No change needed in runtime for this retrain task.

## Minor Logical / Quality Issues Noted (outside scope of this task)

These were observed but **NOT edited** because they are runtime or non-training:

- In old `prep_model.py` (deleted): used BinaryAccuracy + AUC but never macro-F1 for checkpoint.
- Some Android scan logic still had hard-coded 34 in comments (harmless now).
- `server/config.json` had `validation_dir` pointing to non-existent paths (harmless).
- No strong class balancing in very old synthetic generators (fixed in new train.py via weights).

## Recommendations (for after retraining)

- After successful 7-class run, consider a small update to `taxonomy_parser.py` to expose `parent_labels` (pure additive, non-breaking).
- Re-run full server status + one FL round after swapping head to validate.
- Real user data >> any synthetic/COCO for final macro F1.

## Files Touched Summary (this session only)

**Deleted (9 training-only, justified):**
server/scripts/{train_head_bootstrap,retrain_head,build_coco_features,build_gallery_tags,reset_and_retrain}.py
server/{prep_model,prep_eval,coco_to_fgt_mapping,convert_dataset}.py

**Added:**
server/retrain/{config,prepare_dataset,train,evaluate,export_model}.py + README.md
RETRAIN_TRUTH_MAP.md, RETRAIN_CLEANUP_PLAN.md, RETRAINING_GUIDE.md, RETRAIN_AUDIT_NOTES.md

All .md kept, runtime untouched.

**Status:** Repo is now in "set DATASET_DIR → run 4 scripts → get 7-tag head" state.

END AUDIT