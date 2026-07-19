# Aura Gallery — "Liquid Glass" Design & Federated-Learning Development Brief
### GalleryFL Android · grounded in the existing codebase (no re-audit)

**Purpose.** Turn the *Aura Gallery* high-fidelity UI/UX specification (frosted-glass
iOS aesthetic + privacy-preserving Federated Learning) into an implementation-ready
brief for the **existing** GalleryFL Android app. Every claim cites a real file so the
document is auditable against the repo; nothing here re-runs or re-verifies the FL loop
(that was done in prior sessions).

**How to read it.** `EXISTS` = code present today. `PARTIAL` = present but needs work.
`GAP` = not present, needs building. File paths are repo-relative under `android/app/
src/main/java/com/fgt/galleryfl/`.

---

## 1. Existing foundation (the work is mostly done on the engine)

| Layer | Real file(s) | Notes |
|-------|--------------|-------|
| Compose shell | `MainActivity.kt` (`Scaffold`, 3-tab `when(index)` nav: `PhotosScreen`/`ExploreScreen`/`LibraryScreen` + `ImageDetailScreen` overlay) | Functional, not yet `NavHost`-driven with z-axis transitions. |
| Frosted-glass primitive | `ui/components/GlassContainer.kt` | `Modifier.blur(16.dp)` + 24.dp clip, `FGTColors.BgGlass`. **Already the Liquid-Glass atom.** |
| Neumorphic surface | `ui/components/NeuSurface.kt` | Existing soft-surface alternative. |
| Theme | `ui/theme/Theme.kt` (`GalleryFLTheme`, `lightColorScheme`, `dynamicColor=false`, `WindowCompat` status-bar) | Light-only today; edge-to-edge is partial. |
| Type | `ui/theme/Type.kt` | Only `bodyLarge` defined → needs a full iOS-style scale. |
| Palette | `ui/theme/Color.kt` (`FGTColors`) | **Blush / Web3 soft** (rose `#EF4D9B`, lavender `#8B7FE8`, ink-plum text). Conflicts with the spec's iOS System palette — see §2. |
| On-device FL | `data/ml/`: `FeatureExtractor.kt`, `ClassificationHead.kt`, `LocalTrainer.kt`, `DPNoiseInjector.kt`, `GradientClipper.kt`, `PseudoLabelGenerator.kt`, `ThresholdResolver.kt`, `HeatmapGenerator.kt` | Full local training + DP-SGD + GradCAM already exist. |
| Correction loop | `data/local/`: `FeedbackDao.kt`, `FeedbackEntity.kt`, `LocalFeedbackStore.kt`, `RecordTagFeedbackUseCase.kt` | User tag corrections are already captured and stored → feeds local retraining. |
| Organize / Undo | `data/local/`: `OrganizeExecutor.kt`, `OrganizeGalleryUseCase.kt`, `OperationLogEntity.kt`, `OperationLogDao.kt` | Undo-able organization exists (Room `operation_log`). |
| Coordinator connectivity | `data/network/`: `FGTWebSocketClient.kt` (7-arg `onUpdateRequested`, heartbeat), `FGTApiService.kt`, `RetrofitClient.kt`, `TagSignalSender.kt`, `WeightSerializer.kt` | REST + WebSocket weight exchange already implemented. |
| Taxonomy | `data/taxonomy/TaxonomyConfig.kt` (`leafTags`), `TaxonomyFolderPolicy.kt` | 34-tag live taxonomy from server. |
| Local DB | `data/local/AppDatabase.kt` (`@Database` v2, `FeedbackEntity` + `OperationLogEntity`, Room) | **Satisfies the spec's "Room for local tags / smart album metadata."** |
| Server | `server/main.py`, `server/fl_coordinator.py` (trimmed-mean aggregation), `server/security.py` (`validate_update`, `RateLimiter`), `server/tag_demand.py` (demand), `server/prep_eval.py` | Aggregation + demand-weighted taxonomy already live. |

**Takeaway:** the *Federated Learning engine* and a *glass primitive* already exist. The
brief's job is therefore (a) elevate the **UI to premium Liquid Glass**, and (b) close a
short list of **UX/FL gaps** (proactive charging/Wi-Fi training, true Secure Aggregation,
Search tab, motion choreography).

---

## 2. Design-direction reconciliation — DECISION NEEDED

The spec mandates an **iOS System palette** (Background `#FFFFFF`/`#000000`, text
`#000000`/`#8E8E93`, Accent `#007AFF`). The established GalleryFL brand is **Blush / Web3
soft** (rose `#EF4D9B`, lavender `#8B7FE8` on near-white `#FDF6FB`). These are different
brand languages.

**Recommendation (keeps prior hard requirement of brand consistency):** adopt the
*Liquid Glass **material** system* (blur, layering, translucency, motion) from the spec,
but keep **GalleryFL's rose/lavender accent** as the product signature inside the glass.
The spec's exact iOS tokens are supplied below as a **drop-in alternative theme** so the
decision is reversible in one file (`Color.kt`).

| Token | Spec (iOS System) | Recommended (Blush-in-glass) |
|-------|-------------------|------------------------------|
| Surface/base | `#FFFFFF` / `#000000` | `#FDF6FB` (kept) / `#0E0710` |
| Elevated | `#F2F2F7` / `#1C1C1E` | `#FFFFFF` / `#1A1220` |
| Accent | `#007AFF` / `#0A84FF` | `#EF4D9B` / `#B94E8E` (rose) |
| Secondary | — | `#8B7FE8` (lavender) |
| Text | `#000000` / `#8E8E93` | `#2A1B2E` / `#9E2A1B2E` |

> If you want a **pixel-faithful iOS clone**, swap `FGTColors` to the left column and
> change `GlassContainer` blur tint to white. Functionally identical; purely brand.

---

## 3. Part 1 — Visual Identity & Styling (mapping)

| Spec requirement | Existing | Status | Action |
|------------------|----------|--------|--------|
| Frosted glass, 24dp blur, alpha 0.85 layer | `GlassContainer.kt` (16dp) | PARTIAL | Bump `blurRadius` default 16→24dp; add `RenderEffect` explicit path for API 31+ and a 85%-opacity `Box` fallback for <31. |
| Color palette (Light/Dark matrix) | `Color.kt`, `Theme.kt` | PARTIAL | Extend `FGTColors` with the §2 matrix; add a real dark scheme (today `dynamicColor=false`, light-only). |
| Typography: 34sp Large Title → 17sp collapsed, section 20sp, tags 12sp; Inter/SF Pro | `Type.kt` (only `bodyLarge`) | GAP | Add full `Typography` scale + ship **Inter** (`fontFamily` via `Font` resources). Large-title collapse handled in §5. |
| Blur on bottom nav, top headers, context menus | `GlassContainer` reusable | EXISTS | Wrap nav + header + peek menu in `GlassContainer`. |

---

## 4. Part 2 — Structural Architecture (mapping)

| Spec requirement | Existing | Status | Action |
|------------------|----------|--------|--------|
| Edge-to-edge, draw behind status/nav bars (`setDecorFitsSystemWindows(false)`) | `Theme.kt` uses `WindowCompat` but not full draw-behind | PARTIAL | Set `WindowCompat.setDecorFitsSystemWindows(window, false)`; consume insets via `WindowInsets` in `Scaffold` (`padding` + `ime`). |
| Persistent bottom tab bar, 64dp + inset, 24dp blur, 4 tabs (Photos/For You/Albums/Search) | `MainActivity.kt` 3 tabs (Photos/Explore/Library) + `GlassContainer` | PARTIAL | Rename `ExploreScreen`→**For You**, `LibraryScreen`→**Albums**; **add Search tab (GAP)**; float the `NavigationBar` in `GlassContainer`. |
| Bento grid: mixed 1:1 + 4:5 highlights, every 12–15th item spans 2×2 | `PhotosScreen.kt` (uniform grid assumed) | GAP | `LazyVerticalGrid` with `Span` logic driven by a local "aesthetic score" from `FeatureExtractor`/`ClassificationHead`. |
| 2dp gutters, flush-to-edge, 12dp squircle corners | — | GAP | Grid spec in `PhotosScreen`. |
| Live video preview autoplay (silent loop) when >60% in viewport | — | GAP | `AndroidView`/`Video` with `Pager`/scroll-based play; needs a video source path (gallery videos). |

---

## 5. Part 3 — Motion & Micro-interactions (mapping)

| Spec requirement | Existing | Status | Action |
|------------------|----------|--------|--------|
| Pinch-to-zoom grid: 3↔5 cols + Day/Month clustering, `FastOutSlowIn`, alpha-blend during scale | — | GAP | `detectTransformGestures` on `LazyVerticalGrid`; animate `GridCells` count + header grouping; no placeholder jank. |
| Shared-element thumbnail→fullscreen expand; grid dims to black | `ImageDetailScreen.kt` (simple) | PARTIAL | Adopt **`sharedElement`** from Compose `graphics`/`navigation` (or `LookaheadLayout`); dim background to `#000000`. |
| Flick-to-dismiss: 1:1 drag, scale floor 0.75, opacity fade, spring snap (threshold 150dp / 1500dp/s) | — | GAP | `AnchoredDraggable`/`Modifier.draggable` with velocity tracker; spring `DampingRatioLowBouncy`. |
| Long-press Peek & Pop: haptic `EFFECT_CLICK` @200ms, 1.05× lift, 24dp bg blur, vertical action stack (Share/Album/Favorite/Delete/Smart Tags) | — | GAP | `PointerInput` long-press + `Vibrator` + `GlassContainer` bg + bottom-anchored `ModalBottomSheet`. |
| Bottom sheets (radius 24dp, handle) instead of full-screen dialogs | — | GAP | Use `ModalBottomSheet` / `BottomSheetScaffold` for all menus & the Smart-Tags info sheet. |

---

## 6. Part 4 — Smart Tagging & AI UX (mapping)

| Spec requirement | Existing | Status | Action |
|------------------|----------|--------|--------|
| Federated Intelligence Indicator (breathing dual-orbit icon, "Privacy-Engineered Sync Active") | — | GAP | Add to **For You** tab / Settings; reflect `isTraining` from `MainActivity.onUpdateRequested` + `TagSignalSender` activity. |
| Inline Smart Tag chips in detail sheet (32dp pill, ✕ to correct, crumble anim, "Aura learns locally" snackbar) | `FeedbackDao`/`RecordTagFeedbackUseCase` (logic) + `ImageDetailScreen` (no chip UI yet) | PARTIAL | Render `TaxonomyConfig.leafTags` predictions as chips in a swipe-up `ModalBottomSheet`; wire ✕ → `RecordTagFeedbackUseCase`. |
| Real-time semantic search (NL query → Smart Categories + Temporal rows, shared push transition) | No Search screen | GAP | New `SearchScreen` tab (§4) using local metadata + predicted tags; `AnimatedContent` push to filtered grid. |

---

## 7. Federated-Learning Technical Architecture (mapping)

| Spec stage | Existing | Status | Action |
|------------|----------|--------|--------|
| **Local inference** TFLite (MobileNetV3) preloaded, offline auto-tag | `FeatureExtractor.kt` + `models/base_model.tflite` + `ClassificationHead.kt` | PARTIAL | Confirm base model is MobileNetV3-class; expose a `tag(image)` helper used by grid/favorites. |
| **Federated training (idle)** triggered by **charging + Wi-Fi** | Trigger is **server-driven** (`onUpdateRequested`) | GAP | Add `WorkManager` `Constraints(BATTERY_NOT_LOW, requiresCharging, requiresWifi)` to proactively run `LocalTrainer` on corrections; server request remains the global trigger. |
| Compute **gradient updates** locally (NPU/GPU) | `LocalTrainer.kt` (+ `GradientClipper`, `DPNoiseInjector`) CPU gradients | PARTIAL | Keep CPU path; optionally offload to NNAPI/`Delegation` in TFLite Interpreter. |
| **Privacy:** only weights sent, not photos; **Secure Aggregation** | Weights sent via `FGTWebSocketClient`/`WeightSerializer`; aggregation = trimmed-mean (`security.py`) | PARTIAL→GAP | Trimmed-mean is *not* cryptographic SecAgg. Add pairwise-mask / threshold-PAKE style masking if "Secure Aggregation" is a hard requirement (see §8 constraint). |
| **Global update:** server averages → smarter model pushed back | `fl_coordinator.py` aggregation + `main.py` model push + `getCurrentModel` delta | EXISTS | Already wired; surface version in the Intelligence Indicator. |

---

## 8. Implementation Stack & Constraints (mapping)

| Spec stack | GalleryFL today | Status / note |
|------------|-----------------|---------------|
| Kotlin + Jetpack Compose | ✅ `MainActivity.kt`, all `ui/screens/*` | EXISTS |
| `Modifier.blur()` (API 31+) + RenderScript fallback | ✅ `GlassContainer.kt` (`Modifier.blur`) | EXISTS (bump to 24dp; <31 fallback) |
| Jetpack Navigation, custom z-axis push | ⚠️ `when(index)` tabs, not `NavHost` | PARTIAL → migrate to `NavHost` for shared-element + z-anim |
| Accompanist insets (edge-to-edge) | ❌ not in deps | GAP — add `com.google.accompanist:accompanist-insets` (or use core `WindowInsets`) |
| Room for tags / smart albums | ✅ `AppDatabase.kt` | EXISTS |
| **TFF or Flower** for FL client | ❌ custom REST/WS stack | **CONSTRAINT:** prior project rule = *do not replace the custom REST/WebSocket stack with a new framework*. Keep `FGTWebSocketClient`/`FGTApiService`; treat TFF/Flower as out-of-scope unless you explicitly lift that constraint. |
| MobileNet pre-trained base | ✅ `models/base_model.tflite` | EXISTS (verify arch) |

---

## 9. Gap summary

- **EXISTS:** Room DB, GlassContainer blur, full on-device FL + DP-SGD + GradCAM, correction loop (Room), Organize/Undo, WebSocket/REST weight exchange, server aggregation + demand taxonomy, 3-tab Compose shell.
- **PARTIAL:** theme (light-only, no full type scale), edge-to-edge, ImageDetailScreen shared-element, FL trigger (server-driven), LocalTrainer (CPU), aggregation (trimmed-mean, not SecAgg), nav (no NavHost/z-anim).
- **GAP (build):** Search tab + semantic search, Bento grid + pinch-zoom, live video preview, long-press Peek&Pop + haptics, flick-to-dismiss, full bottom-sheet architecture, Intelligence Indicator, Smart-Tag chip UI + swipe-up sheet, proactive charging/Wi-Fi WorkManager trigger, Inter font + type scale, Accompanist insets, dark scheme, true Secure Aggregation (optional).

---

## 10. Phased roadmap (highest leverage first)

1. **Theme & material pass** — §2 palette decision, 24dp `GlassContainer`, full `Type.kt` + Inter, dark scheme, edge-to-edge + insets. *Unblocks everything visual.*
2. **Navigation & shell** — `NavHost`, 4 tabs (add Search stub), floating glass `NavigationBar`, `ModalBottomSheet` everywhere.
3. **Photos tab premium** — Bento `LazyVerticalGrid` + span logic, 2dp gutters, pinch-zoom, live video.
4. **Viewer motion** — shared-element expand, flick-to-dismiss, long-press Peek&Pop + haptics.
5. **Smart Tagging UX** — chip UI + ✕ correction in swipe-up sheet, Intelligence Indicator, Search tab + semantic query.
6. **FL hardening** — WorkManager charging/Wi-Fi trigger, optional NNAPI delegation, optional SecAgg masking.

---

## 11. Verification metrics (spec Part 5) → how to check in GalleryFL

- **Frame velocity (90/120Hz, no jank):** Android Studio Profiler / `Macrobenchmark` on pinch-zoom + sheet expand; assert no dropped frames.
- **Overdraw:** Layout Inspector — black OLED surface drawn only where no image/glass intersects; avoid stacked opaque backgrounds behind `GlassContainer`.
- **Thumb-zone accessibility:** all nav + sheet actions below 60% screen height (Peek&Pop menu anchored bottom). Validate with a layout-bounds pass.
- **Privacy claim:** unit/integration test that `WeightSerializer` payload contains **only** weight tensors (no image bytes) — already structurally true via `FGTWebSocketClient`/`submitUpdate`.

---

## 12. Open decisions for you

1. **Palette:** keep Blush-in-glass (recommended) or pixel-iOS (`#007AFF`)? §2.
2. **Secure Aggregation:** hard requirement (build masking) or is server trimmed-mean + DP-SGD sufficient? §7.
3. **FL framework:** keep the custom REST/WS stack (per prior constraint) or lift it to allow TFF/Flower? §8.
4. **Scope of this pass:** produce the brief only (done), or shall I start implementing Phase 1 (theme/material) or generate the Compose code for a specific component (e.g., `GlassContainer` 24dp + floating glass `NavigationBar`)?
