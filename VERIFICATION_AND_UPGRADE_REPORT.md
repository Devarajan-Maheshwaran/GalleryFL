# GalleryFL / FGT — Verification, Fix & Upgrade Report

**Scope:** Phases 0–6 of the revamp/verification prompt.
**Approach:** Truth map first, then *run the actual core FL loop* with a real (math-mirroring) mock client, trace data flow with evidence, fix only what is broken, and report honestly.
**Environment:** Linux sandbox, Python 3.13 venv with `fastapi`, `uvicorn`, `numpy`, `pydantic`, `httpx`, `websockets`, `pytest`. The server core FL loop does **not** import TensorFlow, so it was run and verified without TF. (Server-side *evaluation* needs TF + a dataset and was not executed — see Blockers.)

---

## Executive Summary

The repository is **substantially real and well-engineered**, not a fake demo. The hardest, highest-risk claim — *does the federated learning loop actually work end-to-end?* — was **verified by running it**: register → download → deserialize → real local training (forward/backward + FedProx + DP clip + DP noise) → serialize → submit → server validate → aggregate → round advance → metrics stored → WebSocket broadcast. **All 8 rounds completed, `model_version` advanced 1→9, and loss converged 0.82→0.62** under DP-OFF.

Two **critical server bugs** that broke multi-round training were found and fixed:
1. **Rate limiter** allowed only 1 update per client per 5 s → blocked every client's round-2+ submission (HTTP 429). Fixed to key on `(client, round)`.
2. **Anomaly detector** computed the norm of the *full weight vector* (global+delta) → ~zero variance across rounds → 3-sigma gate rejected **every legitimate update after round 2**. Fixed to use the **delta** norm.

A third, **design-level** issue was found and flagged: **DP is real but mis-calibrated** — clipping the delta to L2=1.0 across 280 K params and adding per-param Gaussian noise (σ≈4.84/ε) makes noise ~2500× the signal, so the model *diverges* at any usable ε (verified: loss 0.82→~7.0 even at ε=10). Server now emits the DP hyperparameters in `update_requested` so a client can use them; the Android client still hardcodes ε=1.0 (recommended fix documented).

**Decision Gate → Case A (Fixed in place).** No Flower migration. The custom REST+WebSocket protocol is correct and the weight schema is byte-for-byte symmetric between server and Android.

---

## DECISION GATE OUTCOME — "Fixed in place" (Case A)

**Reasoning:** The core FL protocol, weight serialization, aggregation, and Android ML/network layers are correct and verified. The failures found were (a) two targeted server-side bugs (rate limiter, anomaly norm) that are now fixed with ~10 lines each, and (b) a DP calibration/tuning issue that is a design choice, not a structural defect. There is **no irrecoverable schema drift, no fundamental architecture problem, and no mocked gradient path** — the Android `LocalTrainer` runs a genuine forward/backward pass with real features. Migrating to Flower would discard working, verified infrastructure for no correctness gain.

**Effort to reach demo-readiness:** ~1–2 days of focused work (the two critical bugs are already fixed; remaining items are DP tuning, wiring gallery-organization persistence, and populating server-side eval). Flower migration would be a multi-week rewrite with higher risk. Case A is clearly the right call.

---

## PHASE 0 — Repository Truth Map

| # | Area | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Server core (FastAPI, endpoints, config) | **KEEP** | `main.py` defines `/api/register`, `/api/model/current`, `/api/training/submit-update`, `/api/training/{start,stop,status}`, `/api/metrics/*`, `/api/export/*`, `/ws/feed`. Token auth, UDP discovery, heartbeat cleanup all present and correct. |
| 2 | FL coordinator / orchestration | **KEEP + 1 fix** | `fl_coordinator.py` runs full lifecycle, selects online clients, sends `update_requested`, aggregates on `min_clients`, advances rounds, broadcasts events. Only change: added DP params to `update_requested` config. |
| 3 | Model manager / weight state | **KEEP** | `model_manager.py` loads `initial_head_weights.npz`, strict schema validation vs `model_schema.json`, zlib+base64 little-endian serialize/deserialize, delta compute, versioned snapshots. Solid. |
| 4 | Security / update validation | **KEEP + 1 fix** | `security.py` `validate_update` checks layer count, shapes, NaN/Inf, 3-sigma norm. `RateLimiter` and the norm computation were the two bugs — both fixed. `TrustScorer`, `AuditLog` present. |
| 5 | Model pipeline (MobileNetV3, TFLite, taxonomy) | **KEEP** | `prep_model.py` builds `base_model.tflite` with two outputs `[spatial_proj(7×7×1024), pool(1024)]`, extracts head, writes schema. `taxonomy_parser.py` + `taxonomy.json` = 34 leaf classes. `model_schema.json` matches exactly. |
| 6 | Dashboard frontend | **KEEP + minor** | `dashboard.js` is fully driven by real WebSocket events + REST (`round_started/completed`, `client_connected`, `training_complete`, history backfill). Apple-clean palette. Added honest "awaiting evaluation" note for comparison. |
| 7 | Metrics / leaderboard / comparison / privacy / export | **KEEP (comparison needs eval)** | `metrics.py` real leaderboard (images 40% / rounds 30% / acc 30%), summary, report. Comparison reads real eval files — currently `evaluated:false` (no TF run yet). Export model/report work. |
| 8 | Mock clients / automated tests | **IMPROVE** | Original `mock_client.py` was *intentionally removed* for release (RELEASE_CHECKLIST). `tests/` are API/alignment only — **no script runs a full FL round**. I added `verify_fl_loop.py` (verification artifact) and made `test_integration.py` read the live token → 10/10 pass. |
| 9 | Android architecture (single-activity Compose) | **KEEP** | Cleaner than the plan's 7-screen design. `MainActivity.kt` orchestrates Photos/Explore/Library + FL "Model Sync" dialog. No fake/mock/TODO in code (grep-verified). |
| 10 | Android ML engine | **KEEP** | `ClassificationHead` (real forward/backward), `LocalTrainer` (real BCE + **FedProx** `w -= lr*(g + mu*(w−w_global))` + delta clip + DP noise), `FeatureExtractor` (real TFLite), `DPNoiseInjector`, `GradientClipper`, `PseudoLabelGenerator`, `HeatmapGenerator` all implemented and correct. |
| 11 | Android network layer | **KEEP** | `WeightSerializer` is byte-for-byte symmetric with the server. `FGTApiService` header (`X-FGT-Token`) + `ClientUpdateRequest` fields match. `FGTWebSocketClient` matches server WS (heartbeats, reconnect, event routing). |
| 12 | Android gallery organization (albums, EXIF, revert) | **IMPROVE (not wired)** | `AlbumCreator`, `OrganizeGalleryUseCase`, `ExifTagWriter`, `AppDatabase` are implemented but **never called from the UI**. `MainActivity` only builds `smartAlbums` in memory; no disk album creation, EXIF write, or revert is invoked. |
| 13 | Android UI/UX quality | **KEEP (good bones)** | Screens have real state wiring, loading/empty/error handling, themed components (`NeuSurface`, `GlassContainer`, warm palette). Premium, calm aesthetic. Gap: organization persistence not surfaced. |
| 14 | Packaging / run scripts / README / demo | **KEEP** | `run_server.py`, `run_server.sh/.bat`, `scripts/package_server.py`, `README.md`, `RELEASE_CHECKLIST.md` present and coherent. |

---

## PHASE 1 — Core FL Loop Verdict (13 steps)

Ran a real mock client (`server/verify_fl_loop.py`) that mirrors the Android `LocalTrainer` math and drives a full 8-round session over REST+WebSocket. **Output (DP-OFF, reset to initial weights):**

```
[client ...] round 1 -> loss=0.8080 acc=0.0000 submit=200   ...  round 8 -> loss=0.6059 submit=200
=== METRICS HISTORY (8 rounds) ===
  round 1: acc=0.0000 loss=0.8198 clients=4
  round 2: acc=0.0000 loss=0.7830 clients=4
  ...
  round 8: acc=0.0000 loss=0.6201 clients=4
[status] model_version=9
```

| # | Step | Verdict | Evidence |
|---|------|---------|----------|
| 1 | Server loads base_model.tflite + head on startup | **WORKS** | Log: `Loaded model weights v1 ... Layer Shapes: [(1024,256),(256,),(256,34),(34,)]` |
| 2 | Client registers via POST /api/register | **WORKS** | `register_client` broadcasts `client_connected`; curl + mock both get 200 + `client_id`. |
| 3 | Client downloads model via GET /api/model/current | **WORKS** | Returns base64 TFLite/weights; `X-Model-Version` header present. |
| 4 | Client decodes weights, schema match | **WORKS** | Serialize→deserialize round-trip `OK=True` (1.3 MB b64). |
| 5 | Client runs local training (real/simulated) | **WORKS (REAL)** | `LocalTrainer` does forward/backward BCE + FedProx; mock replicated it; loss decreased. |
| 6 | Client serializes updated weights | **WORKS** | `WeightSerializer` symmetric; verified decode on server. |
| 7 | Client submits via POST /api/training/submit-update | **WORKS** | `submit=200` every round after fixes. |
| 8 | Server validates (shape, NaN/Inf, norm) | **WORKS (fixed)** | `validate_update` runs; norm check was on full vector → fixed to delta norm. |
| 9 | Server aggregates at min_clients | **WORKS** | `_trimmed_mean_aggregate` triggered at 4 clients; rounds advanced. |
| 10 | Server updates global model, increments round | **WORKS** | `model_version` 1→9; `update_global_weights` asserts shapes + saves snapshot. |
| 11 | Server broadcasts round completion on /ws/feed | **WORKS** | Clients received `round_completed`/`training_complete` and terminated the session. |
| 12 | Metrics stored & retrievable via /api/metrics/history | **WORKS** | 8 real, non-static entries returned; persisted to `output/metrics_history.json`. |
| 13 | Across rounds, accuracy/loss trends sanely | **WORKS (DP-OFF)** | Loss 0.82→0.62 monotonic. With DP-ON ε=1.0 it diverges (see Aggregation verdict) — a tuning issue, not a loop break. |

> The strict `all-34-classes-exactly-match` accuracy metric reports 0% on synthetic non-IID data; the **loss trend is the definitive convergence signal** and it is clearly sane. The Android uses the same strict metric.

---

## Weight Schema Contract Verdict — **PASS**

Canonical order `w1, b1, w2, b2`; dims `(1024,256)`, `(256,)`, `(256,34)`, `(34,)`; `NUM_CLASSES = 34` everywhere.

| Check | Server (`model_manager.py`) | Android (`WeightSerializer.kt`, `ClassificationHead.kt`) | Match |
|-------|------------------------------|-----------------------------------------------------------|-------|
| Byte order | `flat.astype('<f4')` + `struct.pack('<I', …)` | `ByteBuffer.order(LITTLE_ENDIAN)`, `putFloat` | ✅ |
| dtype | float32 (`'<f4'`) | `Float` (float32) | ✅ |
| Layer length prefix | `struct.pack('<I', len)` (uint32 LE) | `putInt(layerSizeBytes)` (int32 LE) | ✅ |
| Compression | `zlib.compress` + `base64` | `Deflater` + `Base64.NO_WRAP` | ✅ |
| Layer order / indexing | `w1[1024,256]` C-order | `w1[i][j]` input-major `[1024][256]`; `w2[i][j]` hidden-major `[256][34]` | ✅ |
| Symmetric round-trip | `deserialize(serialize(w)) == w` | `serialize(deserialize(...))` equivalent | ✅ (verified `OK=True`) |

`model_schema.json` and `taxonomy.json` (34 leaves) and `TaxonomyConfig.NUM_CLASSES = 34` all agree. **No drift.**

---

## Aggregation Correctness Verdict

| Mechanism | Verdict | Evidence |
|-----------|---------|----------|
| **FedAvg (base)** | **REAL** | `_trimmed_mean_aggregate` averages client weight vectors directly. |
| **Trimmed Mean** | **REAL but conditional** | Activates only when `len(updates) >= 4`: sorts each layer, trims top/bottom `max(1, int(n*trim_pct))`, averages the rest. For 2–3 clients it falls back to a sample-count×trust-weighted average (still a legitimate aggregate, not a last-write-wins overwrite). |
| **FedProx** | **REAL (client-side)** | Server comment: *"FedProx is enforced by the client local objective. Applying it a second time here is not FedProx and double-regularises updates."* Android `LocalTrainer` applies the proximal term: `w -= lr*(grad + mu*(w − w_global))`. Server sends `mu` in `update_requested`; Android uses it. ✅ |
| **Validation before commit** | **SIMPLIFIED** | `update_global_weights` asserts each layer's shape matches the schema (catches shape corruption) but does **not** re-evaluate accuracy on a server-held set. Global accuracy in metrics is the **mean of client-reported** `local_accuracy`/`local_loss` (plausible but not server-verified). This is acceptable for the architecture but should be noted. |
| **Poisoning / outlier defense** | **PARTIAL** | `validate_update` 3-sigma norm gate (now on the **delta**, correctly) + `TrustScorer` weighting in aggregation. Trimmed Mean adds robustness at ≥4 clients. No per-parameter magnitude cap beyond L2 clip. |

---

## PHASE 2 — Build Quality (verified)

- **Server startup:** PASS — starts in ~1 s, loads model, binds 8000, serves dashboard.
- **API wiring:** PASS — all endpoints respond (verified via curl + mock + pytest).
- **WebSocket live updates:** PASS — `round_started/completed`, `client_connected/disconnected`, `update_received`, `training_complete` all broadcast and consumed; dashboard reconnects with backoff; Android sends 15 s heartbeats and reconnects.
- **FL round execution:** PASS — full 8-round session completed (after the two fixes).
- **Model export:** PASS — `GET /api/export/model` returns 1.7 MB TFLite (200).
- **Report export:** PASS — `GET /api/export/report` returns structured JSON (200).
- **Dashboard integrity:** PASS (with honest empty-state for comparison added).
- **Android architecture integrity:** PASS — no fakes; real ML + network + UI wiring.
- **Android UX quality:** PARTIAL — screens are real and themed, but gallery-organization persistence (albums/EXIF/revert) is not wired to the UI.
- **Tests:** PASS — `10 passed` after fixing the hardcoded-token mismatch.
- **Demo readiness:** PARTIAL — loop is demo-ready; DP default and comparison eval need attention (see Blockers).

---

## PHASE 3 — Change Report (what was preserved / fixed / replaced)

**Preserved (good work kept):** entire FL protocol, weight serialization, aggregation, Android ML engine, network layer, dashboard, model pipeline, taxonomy, packaging.

**Fixed (smallest-layer root-cause fixes):**

1. `server/security.py` — `RateLimiter` now keys on `(client_id, round)` instead of a fixed 5 s wall-clock window. *Why:* the old limiter blocked every client's 2nd+ round submission (HTTP 429) because fast clients finish rounds in <5 s, so multi-round FL was impossible. *Before:* `if now-last<5: reject`. *After:* anti-replay per (client, round) with a 2 s window, allowing legitimate successive rounds.
2. `server/main.py` — `submit_update` now computes `update_norm` from the **delta** (`weights − global_weights`) instead of the full submitted vector. *Why:* the full-vector norm is dominated by the near-invariant global head (≈27.97, σ≈0) so the 3-sigma gate rejected every update after round 2 ("Norm 27.98 exceeds 3-sigma (27.97 + 3*0.00)"). *After:* the delta norm is the meaningful anomaly signal.
3. `server/tests/test_integration.py` — token now read from `config.json` (was hardcoded `fgt-pass`, mismatching the real `2c79bdfe`). 10/10 tests pass.
4. `server/fl_coordinator.py` — `update_requested` now includes `dp_epsilon`, `dp_delta`, `max_grad_norm` so the server config is authoritative for client-side DP.
5. `server/main.py` `get_comparison` — returns `"evaluated": false` when no server-side eval has run, instead of implying 0% accuracy.
6. `server/dashboard/{dashboard.js,index.html}` — comparison panel shows an honest "awaiting evaluation" note when `evaluated` is false.

**Added:** `server/verify_fl_loop.py` — a real, math-faithful mock FL client used to prove the loop (kept as a verification artifact; not part of the release zip).

**Replaced:** nothing of substance — no working code was discarded.

---

## PHASE 4 — Dashboard UI/UX Upgrade Report

**Usability:** The dashboard is a single-page console driven entirely by live WebSocket events + REST backfill. Start/Stop, round progress, client cards (with update-received highlight), leaderboard, accuracy/loss charts, privacy diagram, and export modal are all functional. No dead buttons; export buttons correctly enable only after a session; copy/regenerate-token work.

**Visual:** Clean Apple-inspired palette (blue/gold/indigo on light gray), neumorphic-style raised cards, smooth Chart.js transitions, animated WS status dot, modal transitions, sidebar scroll-spy. Calmer and more "product" than the warm neumorphic spec — **kept and evolved, not replaced**.

**Interaction improvements made:** honest empty/awaiting state for the FL-vs-baseline comparison (prevents a misleading all-zero chart), consistent with the "no fake/static labels" requirement.

**Remaining (non-blocking):** The comparison/F1 charts render zeros until a server-side evaluation runs (`prep_model.py evaluate` needs TF + a labelled dataset, not present here). This is **real absence of data, not fake data** — left honest rather than fabricated.

---

## PHASE 5 — Android UI/UX Upgrade Report

**Usability:** `MainActivity` wires a complete, real flow: connect (parse `IP:PORT@TOKEN`) → register → download model → on `update_requested` extract features, pseudo-label, run `LocalTrainer` (FedProx + DP) → submit; Explore scans recent images through the head to build smart albums; Library browses them; ImageDetail supports delete. State (connected/training/round/version) is real and reactive.

**Visual:** Warm palette (`BgBase #F0E4D7`, terracotta/gold accents), `NeuSurface`/`GlassContainer`, rounded search, floating bottom nav, training spinner in the profile button — a calm, premium gallery feel.

**Interaction:** Permission gate, smooth tab transitions, search by tag/album, scan progress + error toasts, FL sync dialog with connect/disconnect. Good empty/loading/error states.

**Gap (recommended next step, not a bug):** Gallery **organization persistence** is implemented in `AlbumCreator`/`OrganizeGalleryUseCase`/`ExifTagWriter` but **not invoked** from the UI — `MainActivity` only computes in-memory `smartAlbums`. To reach the plan's "Albums + EXIF + Revert" MUST-HAVE, wire an "Organize" action that calls `OrganizeGalleryUseCase` (create `Pictures/FGT/<tag>/` copies, write EXIF `USER_COMMENT`, log to `AppDatabase`, support revert). This is the single biggest remaining UX feature.

---

## PHASE 6.9 — Build Quality Report (summary table)

| Check | Result |
|-------|--------|
| Server startup | **PASS** |
| API wiring | **PASS** |
| WebSocket live updates | **PASS** |
| FL round execution | **PASS** (after fixes) |
| Model export | **PASS** |
| Report export | **PASS** |
| Dashboard integrity | **PASS** |
| Android architecture integrity | **PASS** |
| Android UX quality | **PARTIAL** (org persistence not wired) |
| Demo readiness | **PARTIAL** (see blockers) |

---

## Remaining Blockers / Recommendations

| # | Issue | Impact | Blocks demo? | Recommended next step |
|---|-------|--------|--------------|------------------------|
| B1 | **DP mis-calibration** (delta clipped to L2=1.0 over 280 K params + Gaussian σ≈4.84/ε → noise ~2500× signal; model diverges at any usable ε). | Real but breaks convergence with default ε=1.0. | No (loop works; only utility degrades). | Apply DP-SGD at the **per-example gradient** level during local training (clip per-example to `max_grad_norm`, average, then add noise), and/or raise the effective ε. Server already sends `dp_epsilon/dp_delta/max_grad_norm` — **make the Android `DPNoiseInjector` read them** (currently hardcoded 1.0). |
| B2 | **Gallery-organization persistence not wired** (`AlbumCreator`/`ExifTagWriter`/`OrganizeGalleryUseCase` unused). | Feature exists in code but invisible to users. | No (core FL + tagging work). | Add an "Organize" action in `MainActivity`/`ExploreScreen` invoking `OrganizeGalleryUseCase`; surface revert from `AppDatabase`. |
| B3 | **Comparison/F1 charts need server-side eval** (`output/*.json` absent; no TF/dataset here). | Charts show honest "awaiting evaluation", not fake numbers. | No (dashboard otherwise live). | Run `prep_model.py bootstrap_train` + `evaluate` on a labelled set to populate `output/latest_eval.json` & `output/bootstrap_eval_report.json`. |
| B4 | **Server-side validation of accuracy** uses client-reported metrics (mean), not a held-out eval. | Metrics are plausible but not independently verified. | No. | Optionally run `_run_evaluation` (already wired) and store server-computed accuracy. |
| B5 | **`HeatmapGenerator` / `GradCAM`** appears unused in the shipped UI flow. | Nice-to-have explainability not surfaced. | No. | Wire a "why this tag" heatmap into `ImageDetailScreen`. |

**Confidence:** High for everything verified by execution (server loop, schema, aggregation, dashboard serving, tests). The Android training path is verified by **static trace + byte-symmetric serialization + a server-side run driven by an identical math client**; it was not compiled/run on a device here (no Android SDK/emulator), so on-device rendering is assessed by code review only.

---

## How to reproduce the verification

```bash
cd server && python -m venv .venv && . .venv/bin/activate
pip install fastapi "uvicorn[standard]" numpy pydantic httpx websockets pytest pytest-asyncio
uvicorn main:app --port 8000 &
python verify_fl_loop.py            # DP-ON (shows loop works; diverges under ε=1.0)
python verify_fl_loop.py --no-dp    # DP-OFF: loss converges 0.82 -> 0.62 over 8 rounds
python -m pytest tests/ -q          # 10 passed
```

**Bottom line:** The project is a genuine, working federated-learning gallery app. The critical loop bugs are fixed, the schema is proven symmetric, and the remaining work is tuning (DP) and feature-wiring (gallery org persistence, eval populating) — all within Case A, none requiring a framework rewrite.
