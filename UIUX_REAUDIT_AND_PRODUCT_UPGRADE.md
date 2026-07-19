# UI/UX Re-Audit & Product Upgrade — GalleryFL

Scope: **frontend dashboard**, **Android UI/UX**, **shared theme**, and
**low-level product features** (token flow, dynamic tags/taxonomy, connection,
admin polish, empty/loading/error states). The backend core FL architecture was
**not** touched (only two small read-only endpoints were added).

Verification stance: the **dashboard was verified live** against the running
server (endpoints return the expected shapes; JS passes `node --check`; the new
HTML/CSS/JS are served). **Android was code-audited and edited but not compiled
/ run** — no Android SDK or emulator is available in this environment. Where a
claim is code-only, it is stated as such.

---

## 1. What was audited

### Dashboard (live, against running server)
- REST + WebSocket bindings: `/api/training/status`, `/api/metrics/*`,
  `/api/training/start|stop`, `/api/training/regenerate-token`,
  `/api/export/*`, `ws/feed`. All consumed by the new UI.
- Every panel is fed by real backend data. **No hardcoded metrics remain.**
- Prior issue: the dashboard pulled **Chart.js from a CDN** — broken in the
  offline in-app preview and contrary to the "self-contained" requirement.
  Replaced with a **self-contained canvas chart engine**.
- Prior bug: `styles.css` placed `#panel-accuracy` / `#panel-loss` in the
  2-column grid, but the HTML used `panel-metrics` / `panel-loss` → those cards
  were mis-placed. Fixed in the rewrite.
- Theme was "Apple glass / blue" — visually disconnected from Android's
  "dark midnight / indigo". Not a unified product family.

### Android (code audit)
- IA: 3-tab bottom nav (Photos / Explore / Library) + photo detail — already
  gallery-shaped; confirmed "photos first".
- Found: `FGTColors` is a **dark** palette while the dashboard is light → the
  two surfaces used opposite color logic.
- `Theme.kt` enabled `dynamicColor` (Android 12+ would override the brand
  palette with wallpaper colors).
- `FLSyncDialog` had **no input validation** and did not remember the code.
- `PhotosScreen` had **no empty state**.
- Organize/Undo and GradCAM wiring (added previously) were confirmed intact.

---

## 2. What was changed

### Dashboard — full rewrite (`index.html`, `styles.css`, `dashboard.js`)
- **Design system**: light "Blush / Web3 soft" — near-white pink surfaces, rose
  primary + lavender secondary, soft pink-tinted shadows, rounded squircles.
- **Self-contained charts** (no CDN): accuracy + loss lines, baseline-vs-
  federated grouped bars, per-class F1 horizontal bars.
- **Admin clarity**: Overview now exposes phase, round, model version,
  parameters, registered/online counts; Training panel shows the live
  aggregation mode and DP config; a dedicated **Privacy** card shows the live
  DP (ε, δ, clip) and aggregation mode from the server.
- **Control surface**: Start/Stop with disabled/loading states and live round
  progress; config inputs; export model/report behind a confirm modal (enabled
  after a session).
- **Token flow (product-grade)**: access code is **masked by default** with
  Show/Hide, one-tap Copy, and Regenerate with explicit safety copy
  ("regenerating revokes every connected client"). All actions fire toasts.
- **Empty / loading / error states** everywhere (clients, leaderboard,
  taxonomy, charts "awaiting evaluation", toasts for failures).
- Removed the stale `#panel-*` placement bug.

### Android — theme + flow (code-only)
- `Color.kt`: flipped the **entire app** to the shared light pink/white
  palette by redefining the existing `FGTColors` constants (no per-screen
  edits needed). One-file change re-themes every screen.
- `Theme.kt`: `dynamicColor = false` so the brand palette is enforced on
  Android 12+; defaults to light.
- `FLSyncDialog`: added **format validation** (`IP:PORT@ACCESS_CODE`), inline
  error text, and **remembered code** (persisted to `fgt_prefs`, pre-filled on
  next open) for smoother reconnect.
- `PhotosScreen`: added an on-brand **empty state**.
- Organize/Undo buttons (Explore) and GradCAM toggle (detail) retained.

### Backend — two read-only endpoints (verified live)
- `GET /api/taxonomy` → live taxonomy (categories + leaf tags).
- `GET /api/config` → live training/privacy knobs (ε, δ, clip, lr, μ, trim,
  aggregation). Used by the dashboard's Privacy/Training panels so the operator
  sees the real coordinator configuration.

### Shared language
- New `DESIGN_SYSTEM.md` defines the unified palette, type, shape, components,
  and naming table. Dashboard CSS variables and Android `FGTColors` use the
  same hex values.
- Naming aligned: "Coordinator" (both), "Training" (dashboard) / "Model Sync"
  (Android, consumer-friendly synonym), "Taxonomy" (tags), "Privacy/DP".

---

## 3. Verification: live vs code-only

**Verified live (running server):**
- `/api/config` → `dp_epsilon 1.0, clip 1.0, agg trimmed_mean, classes 34`.
- `/api/taxonomy` → 7 categories, 34 tags.
- `/api/metrics/comparison` → `evaluated: true`, 7 categories,
  baseline ≈ `0.476` → federated ≈ `0.512` (federated above baseline in all 7).
- `/api/metrics/leaderboard`, `/api/metrics/history`, status, export, token —
  all reachable and well-shaped.
- `dashboard.js` passes `node --check`; new HTML/CSS/JS served (200).
- Endpoint list consumed by the dashboard matches the server exactly.

**Code-only (cannot build/run here):**
- Android theme flip, connect-flow validation/persistence, empty state,
  Organize/Undo + GradCAM wiring. Verified by reading the code and confirming
  constant/import usage; not compiled. A Gradle build on a machine with the
  Android SDK is the remaining confirmation step.

---

## 4. Before → after (major decisions)

| Area | Before | After |
|---|---|---|
| Charts | Chart.js via CDN (offline-broken) | Self-contained canvas engine |
| Theme | Dashboard light-blue glass vs Android dark indigo (two styles) | One Blush/Web3 soft system, same hex both sides |
| Token | Plain code + copy; no masking/safety | Masked + Show/Hide + Copy + Regenerate w/ safety copy + toasts |
| Taxonomy | Hardcoded in UI | Live from `/api/taxonomy` |
| Privacy | Static SVG only | Live DP (ε/δ/clip) + aggregation from `/api/config` |
| Android connect | No validation, no memory | Validated format + remembered code |
| Empty states | Missing (Photos) | On-brand empty/loading/error everywhere |

---

## 5. Remaining limitations / gaps

- **Android not compiled** in this environment — the theme flip and flow edits
  are validated by inspection, not by a build. Risk is low (constant names and
  APIs are unchanged) but a build is the final gate.
- **Dynamic / demand-driven tags**: the taxonomy is now *live from the server*
  (not hardcoded in the UI), but there is still **no backend signal for
  emergent or demand-weighted tags** — the taxonomy is server-authored. Wiring
  demand (e.g., from client feedback or evaluation) would need a backend change
  beyond the read-only endpoints added here; left as a recommended follow-up.
- Android **dark mode** is intentionally dropped in favor of the unified light
  system; if a dark console is later desired, it should be a second, explicitly
  themed variant of the same tokens — not a free dynamic-color override.
- The dashboard privacy SVG is illustrative; the *numbers* beside it are live.

---

## 6. Files changed

Dashboard
- `server/dashboard/index.html` (rewrite)
- `server/dashboard/styles.css` (rewrite — design system)
- `server/dashboard/dashboard.js` (rewrite — live bindings + canvas charts)

Backend (read-only additions)
- `server/main.py` — `GET /api/taxonomy`, `GET /api/config`

Android
- `android/.../ui/theme/Color.kt` (light palette)
- `android/.../ui/theme/Theme.kt` (dynamicColor off, light default)
- `android/.../MainActivity.kt` (FLSyncDialog validation + remembered code;
  Organize/Undo wiring retained)
- `android/.../ui/screens/PhotosScreen.kt` (empty state)

Docs
- `DESIGN_SYSTEM.md`
- `UIUX_REAUDIT_AND_PRODUCT_UPGRADE.md` (this file)
