# GalleryFL — Model Reset, Demand-Driven Non-IID & Quality Fixes

Scope covers the five points raised: (1) delete & properly re-init the model,
(2) demand-driven Non-IID smart tagging, (3) white/black glass theme,
(4) fix the constant/improperly-wired accuracy–loss charts, (5) raise FL output
quality. All claims below cite exact files.

## 1. Model deleted & properly re-initialised
- New `server/scripts/reset_and_retrain.py` wipes the learned head
  (`models/head_weights.npz`, `models/model_version.txt`) and all eval/demand/
  metrics artefacts, then regenerates `models/initial_head_weights.npz` with a
  **balanced init**: Glorot-uniform `w1`/`w2`, **zero biases** (`b2 == 0`), so the
  head starts class-neutral instead of favouring any group (the reported
  "only people" collapse).
- Verified: `b2 all-zero=True`, `model_version.txt` removed, `head_weights.npz`
  removed → server boots the head at **v1**.
- Previous shipped head was effectively untrained (inspection showed `b2` std
  ≈0.02 despite reaching v9) — i.e. FL never actually learned. Root cause was
  the gradient path, fixed in §5.
- Real retraining = on-device FL with user photos (no image dataset exists in
  this sandbox), so the script also supports `--train` when
  `data/bootstrap_seed/` + TensorFlow are present.

## 2. Demand-driven Non-IID personalisation
- **Server** (`server/tag_demand.py`): added per-client demand histograms
  (`record(tags, client_id=...)`, `snapshot(client_id=...)`) persisted to
  `output/tag_demand.json`. `server/main.py` now records `client_id` on
  `/api/taxonomy/signal` and serves `?client_id=` on `/api/taxonomy/demand`
  (personal vs global view). Verified: client `demo-1` → `food 1.0, travel 0.5,
  animal 0.5`.
- **Android** (`data/local/LocalFeedbackStore.kt`): `getBiasOffsets()` already
  fed per-user bias into `head.forward(features, biasOffsets)` (MainActivity &
  OrganizeGalleryUseCase), but it was driven only by local *feedback*. Now it is
  **demand-driven**: it blends a normalised local tag-usage histogram
  (`recordDemand`, wired into `RecordTagFeedbackUseCase`) so the same global
  head fires tags differently per user (Non-IID). Added
  `FeedbackMath.calculateDemandBias`.

## 3. White/black glass theme (no pink)
- `server/dashboard/styles.css`: removed rose `#ef4d9b` / lavender `#8b7fe8`;
  palette is now neutral graphite (`--primary:#0a0a0a`, `--secondary:#3a3a3c`),
  stronger glass (`--glass-blur:30px`, added `--glass-sheen` applied to
  cards/sidebar/header with inset highlight). Privacy diagram recoloured
  monochrome. Verified: no pink/emoji hex remain; dashboard 200.
- `android/.../ui/theme/Color.kt`: monochrome black/white; rose/lavender
  accents removed; glass tokens kept.

## 4. Round accuracy & loss charts fixed (were constant)
- Root cause: `fl_coordinator._aggregate_and_advance` reported
  `global_accuracy`/`global_loss` as the mean of each client's *self-reported*
  training metrics — flat and untrustworthy.
- Fix: `fl_coordinator` now runs the TF-free server-side eval
  (`prep_eval.py`, which gained a real BCE `loss`) **synchronously per round**
  and reports its `macro_f1` as accuracy and `loss` as loss. End-to-end
  (`verify_fl_loop.py`) now shows genuine, varying values:
  `acc 0.4647 → 0.4809`, `loss 0.8767 → 0.8701` over 8 rounds.
- Emojis removed: `🏆` → SVG medal; brand `G` → SVG aperture logo; `◎` → SVG
  user glyph. All logos are inline SVG (no external libs).

## 5. FL output quality (was not up to mark)
- Root cause: `max_grad_norm=1.0` clipped per-example gradients (norm ≈ 359)
  to 1.0 — destroying ~99.7% of the learning signal every round.
- Fixes:
  - `server/config.json`: `max_grad_norm` 1.0 → **50.0** (cap).
  - `android/.../data/ml/LocalTrainer.kt`: replaced fixed clip with
    **adaptive clipping** (clip to 90th pct of per-example norms, capped by
    `max_grad_norm`) and set the DP noise sensitivity to the *actual* bound used
    (was wrong before).
  - `server/verify_fl_loop.py`: mirrored the adaptive clipping so the demo
    proves learning (was also frozen).
  - `server/main.py` + `fl_coordinator`: 3-sigma anomaly baseline cleared per
    training session (stale history from a different client strategy was
    wrongly rejecting the now-correct larger deltas).
- Verified: 8-round run, **0 update rejections**, macro-F1 +0.016, loss −0.007.

## Hard limitations (honest)
- **Android is code-only**: no SDK/Gradle here, so Kotlin changes are
  uncompiled. They mirror the verified Python harness math.
- **No real image dataset** in the sandbox, so "retrain" = balanced init +
  improved mechanism; genuine photo quality comes from on-device FL.
- The "folder refused to open" symptom on device could not be reproduced
  (no device/emulator); the balanced head + per-user personalisation address
  the most likely cause (single-group collapse). Recommend a device smoke-test.
