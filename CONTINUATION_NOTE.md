# GalleryFL — Continuation Note (session after UI/UX re-audit)

Scope of this session: re-establish a **verified live baseline** (the prior session's
"verified live" state was lost when this sandbox was restored) and complete the one
feasible "Not Solved" item from the prior audit — **demand-weighted / dynamic taxonomy**.

---

## 1. Environment discoveries (important — these change what is possible here)

The Arena session snapshot excludes `.venv` **and** `.git/config` / credentials. On
restore this produced two real blockers:

- **Server venv was gone** → rebuilt from `server/requirements.txt`
  (`python3 -m venv .venv && pip install -r requirements.txt`, Python 3.13.13).
  All imports verified. Server restarted and serving on `:8000`.
- **Git remote is gone** → `git remote -v` is empty and `.git/config` does not exist,
  so **pushing to GitHub is impossible from this sandbox** unless the remote URL +
  credentials are re-supplied. All work remains local/uncommitted.
- **Stale-server confusion** → port `:8000` was held by a pre-edit instance of the
  server (started before this session's `main.py` edits), which is why the first
  restart returned 404 for the new endpoints. After killing it by PID, a fresh server
  with the new code binds correctly.
- **Arena infra server** runs as root on `:49999` (`/root/.server/.venv/.../uvicorn
  main:app --port 49999`). It is unrelated to GalleryFL and must not be touched.

---

## 2. Re-verified live baseline (this environment)

| Check | Result |
|-------|--------|
| Server up (`GET /api/config`) | `200` — `dp_epsilon 1.0, dp_delta 1e-05, max_grad_norm 1.0, aggregation trimmed_mean, num_classes 34` |
| `GET /api/taxonomy` | `200` — 7 categories, 34 tags |
| `GET /dashboard/` | `200` |
| Integration suite | **7/7 pass** (`pytest tests/test_integration.py`) |

---

## 3. New feature: demand-weighted taxonomy (the "dynamic tags" product item)

Previously the taxonomy was static (served from `taxonomy.json`). There was no
backend signal for *which tags users actually use*. This adds a recency-weighted
demand signal, additive to the FL core.

### 3a. Server (verified live)
- **New `server/tag_demand.py`** — thread-safe store with recency decay
  (`count *= decay` per `record()`) and durable JSON persistence to
  `output/tag_demand.json`. Unknown/corrupt files degrade to empty instead of crashing.
- **`server/main.py`** — two new endpoints, both consistent with existing auth:
  - `POST /api/taxonomy/signal` (requires `X-FGT-Token`) — body
    `{"client_id": str, "tags": [str]}`. Validates tags against the live taxonomy
    leaves; unknown tags are ignored. Returns `{"ok", "received", "recorded", "client_id"}`.
  - `GET /api/taxonomy/demand` — returns `{"updated_at","total_signals",
    "num_active_tags","weights":{tag:{count,weight}},"trending":[{tag,count,weight}]}`.

Live evidence (signals POSTed, then read back):
```
POST /api/taxonomy/signal  {"tags":["beach","food","not_a_real_tag"]}
  -> {"ok":true,"received":3,"recorded":2}          # invalid tag dropped
GET  /api/taxonomy/demand
  -> {..., "total_signals":7, "num_active_tags":5,
       "trending":[{"tag":"food","weight":1.0},{"tag":"beach","weight":0.98},
                   {"tag":"selfie",...},{"tag":"mountain",...},{"tag":"city",...}]}
```
The decay is visible (repeated `beach`/`food` settle just under `1.0`; one-shot
tags sit lower) — the signal is genuinely *dynamic*, not a lifetime tally.

### 3b. Dashboard (verified served)
- **`server/dashboard/index.html`** — new "Trending" nav item + a `Trending Tags`
  card (`#panel-demand`) between Taxonomy and Export.
- **`server/dashboard/styles.css`** — `.demand*` Blush-themed bars (rose→lavender
  gradient), full-width grid span at ≥1100px.
- **`server/dashboard/dashboard.js`** — `refreshDemand()` fetches
  `/api/taxonomy/demand`, renders ranked bars with an empty state
  ("No demand signals yet…"); wired into boot + the 5s poll. `node --check` passes.

### 3c. Android client emitter (code-only — NOT compiled here)
No SDK/Gradle in this sandbox, so this follows the project's established
code-audit-only workflow and needs a Gradle build on a machine with the Android SDK.
- **`FGTApiService.kt`** — `postTagSignal(...)` + `TagSignalRequest` / `TagSignalResponse`.
- **New `TagSignalSender.kt`** — singleton configured once at connect time with the
  live API service / token / client id; `emit(tags)` is fire-and-forget
  (`Dispatchers.IO` + `SupervisorJob`, all exceptions swallowed) so it can never
  block or crash the UI.
- **`MainActivity.kt`** — `TagSignalSender.configure(...)` right after
  `apiService.register(...)` (uses the existing wildcard
  `import com.fgt.galleryfl.data.network.*`).
- **`ImageDetailScreen.kt`** — on a high-confidence prediction, emits the predicted
  tag (`TagSignalSender.emit(listOf(predictedName))`).

### 3d. Test
- **`server/tests/test_integration.py`** — `test_tag_demand_signal_and_read`: asserts
  401 without token, `recorded == 2` for `[beach, food, not_a_real_tag]`, and that the
  demand read returns `beach`/`food` in `trending`. Suite is now **7/7**.

---

## 4. What remains limited

- **Android not compiled** — the emitter is code-audited only; final gate is a Gradle
  build on a machine with the Android SDK.
- **Git push blocked in this sandbox** — no remote/credentials after snapshot restore.
  To push, re-add the remote (`git remote add origin <url>`) and supply creds, then
  commit. 18 modified + 9 untracked files are ready.
- **Demand Android emitter unverified at runtime** — verified only by server-side
  simulation (curl) of the exact request shape the client sends.

## 5. Before / after (demand feature)

- **Before:** taxonomy was a static list; the dashboard showed every tag with equal
  weight; clients had no way to influence what "matters".
- **After:** clients report the tags they use; the server ranks them by recency-
  weighted demand; the dashboard surfaces a live **Trending Tags** view. This makes
  the product feel alive and lays groundwork for demand-biased taxonomy ordering —
  without touching weights, aggregation, or the WebSocket protocol.
