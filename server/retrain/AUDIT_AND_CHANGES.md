# Retrain truth map and surgical changes

## Truth map at the start of this fix

| Item | Truth | Action |
|---|---|---|
| `prepare_dataset.py` | Basic per-class split was directionally correct, but it allowed empty class splits, absolute-only manifests, duplicate leakage, and missing classes. | Rewritten with deterministic non-empty stratified splits, portable paths, and duplicate/conflict checks. |
| `train.py` | Used Keras and class weights, but standardized train/validation features and then exported only the unfused dense weights. Runtime received raw features, so the deployed graph did not match training. It also promoted a head before held-out evaluation. | Rewritten as one standard Keras `fit` loop with class-weighted cross-entropy, validation argmax macro-F1 checkpointing, patience-based stopping, and no premature deployment. |
| `evaluate.py` | Buggy: evaluated raw features with a head trained on standardized features. `--dataset-dir` was ignored. | Rewritten; loads checkpoint normalization, uses argmax only, reports all seven classes, proves folded export equivalence, and writes the FL test-feature cache. |
| `export_model.py` | Copied the incompatible normalized-space weights directly to runtime and did not update the 34-class schema. | Rewritten; folds normalization into `w1/b1`, validates shapes/numerics, atomically writes the exact four runtime tensors, and updates seven-class schema/version/threshold metadata. |
| `config.py` | Seven labels were correct, but path and output contracts were loose. | Rewritten as the pipeline source of truth. |
| `thresholds.json` | No tracked or generated file was present. Earlier text claimed that fixed `0.5` thresholds were safe for argmax, which is false: argmax does not use thresholds. | Export now writes explicit single-label/argmax metadata. Optional values are post-argmax abstention gates only. |
| Metrics reports | No real new held-out report was in the repository. `bootstrap_metrics.json` was obsolete 34-leaf binary-accuracy/AUC history and could not support the requested comparison. | Obsolete report removed. Evaluation generates accuracy, macro metrics, per-class metrics, confusion matrix, and an honest old/new comparison. |
| Frozen backbone integration | Server model had `[1,1024]` and `[1,7,7,1024]` outputs. Android shipped a different one-output `[1,960]` model while allocating 1024 floats, and its two-output map was reversed for the server model. | Canonical server backbone copied to Android; Android now discovers outputs by shape. |
| Class contract | Training said seven parents while model schema, fallback head, server taxonomy count, Android taxonomy count, and several tests remained 34 leaves. | Runtime contract is now seven parent classes everywhere; the detailed 34-leaf taxonomy remains available as metadata. |
| FL round evaluation | `fl_coordinator.py` called deleted `prep_eval.py`. | Replaced with framework-free evaluation over `test_features.npz`, with honest client-metric fallback if no held-out cache exists. |

## Removed as obsolete

- Redundant root retrain audit/plan/summary documents that described deleted
  34-leaf scripts or claimed unverified scores.
- `server/models/bootstrap_metrics.json`, a 34-leaf multi-label training history
  with no macro F1 and no compatible held-out report.

No server main module, model manager, Android runtime, base model integration,
FL serialization path, or required model artifact was deleted.

## Architecture and semantics

- Frozen canonical GalleryFL TFLite backbone.
- 1024-D projection -> Dense(256, ReLU) -> Dense(7 logits).
- Dropout during centralized training only.
- Class-weighted sparse categorical cross-entropy.
- Validation checkpoint and early stopping on seven-class macro F1.
- Held-out test accuracy/F1 computed with one `argmax` per image.
- Android local head now uses softmax/categorical semantics and one-hot top-1
  pseudo-labels, matching the centralized seed.

## Measured held-out result

Training was run against the Kaggle **COCO Minitrain 10K** YOLO dataset using
the explicit proxy map in `prepare_dataset.py`. COCO train2017 supplied 8,501
train and 1,499 validation examples; untouched COCO val2017 supplied 4,952 test
examples. The best checkpoint was epoch 38 by validation macro F1.

| Metric | Old | New held-out test |
|---|---:|---:|
| Macro F1 | 0.0330 | **0.3055** |
| Accuracy | not present in old report | **0.3407** |

Per-class test F1: people 0.3208, places 0.3664, activities 0.3742, objects
0.3769, documents 0.1516, nature 0.3945, events 0.1541. Full precision, recall,
confusion matrix, supports, and exact values are in
`server/output/retrain/test_metrics.json`.

The +0.2725 macro-F1 improvement clears the enforced 0.20 export gate. Historical
accuracy is deliberately `null` because it was not present and is not invented.
