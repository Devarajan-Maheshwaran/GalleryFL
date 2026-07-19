# GalleryFL Repo Truth Map (HEAD 5a966ed + current files, 2026-07-19)

## What ACTUALLY Works (evidence from code + execution)
- Server boots and serves: `uvicorn main:app` + `/api/training/status` returns 200 with real data (model_version, registered=0, access_code containing real 8-char token "P54RCJHM").
- Token system: 8-char alphanumeric (security.py:generate_access_code + AccessCodeGenerator.kt) using identical alphabet (no 0/O/1/I/L). Verified match.
- Weight serialization: symmetric (server ModelManager + Android WeightSerializer). Schema: 4 layers w1/b1/w2/b2, 34 classes consistent (model_schema.json + taxonomy).
- Model load: server loads head_weights.npz or initial on startup. Android ModelStateStore persists/restores weights + version (binary little-endian format).
- Register: `/api/register` (X-FGT-Token header) works, returns client_id + model_version. Idempotent by client_id.
- Model fetch: `/api/model/current` (full or delta via client_version). Headers X-Model-Format / Version.
- Update submit: `/api/training/submit-update` path exists (auth, rate, validate, coordinator).
- WS: `/ws/feed` + events (update_requested, round_*, client_*, training_complete). Heartbeats. Android FGTWebSocketClient has reconnect + heartbeat.
- Coordinator: registers, round lifecycle, submit logic (stale/duplicate/version checks), aggregation (trimmed mean + demand + trust).
- Dashboard: real WS-driven + REST backfill. Charts, clients, leaderboard, status live.
- Android gallery/scan: Photos, Explore (smartAlbums from ClassificationHead forward), Search, permission, search filter. In-memory scan + pseudo labels + trainer.
- Android persistence (partial): ModelStateStore (weights/version), ScanResultDao (albums survive), OperationLog (undo), MediaState (fav/archive/trash), AppDatabase v3.
- Organize: OrganizeExecutor + AlbumCreator + Exif + UseCase fully implemented and *wired* to ExploreScreen buttons (onOrganizeClick, onUndo).
- LocalTrainer: real forward/BCE + FedProx + clip + DP noise.
- Config single source-ish (server/config.json + ServerConfig).
- Server starts with LAN IP + prints token.

## What is PARTIALLY WIRED
- Connection state: last_access_code saved in prefs. Model + scan results restore on launch. But *full session* (serverUrl, token, registeredClientId, isConnected) is only in-memory Compose `remember`. No auto-resume of FL connection.
- Scan persistence: works for smartAlbums via DB + restoreScanResults.
- WS reconnect: Android has logic; server has on_client_reconnected (re-issues round if eligible). But registration state lost on app kill.
- Tag signal / demand: wired on register path, /api/taxonomy/* present.
- Model restore in scan: checks activeWeights + /api/model/status.

## What is BROKEN / Real Errors (evidence)
1. **Token UX (major violation of critical req):** 
   - parseConnectionDetails requires "IP:PORT@TOKEN" split.
   - FLSyncDialog: placeholder "IP:PORT@ACCESS_CODE", validation `parts.size != 2`, error "Use the format IP:PORT@ACCESS_CODE".
   - Dashboard: access_code = "IP:PORT@TOKEN", mask shows IP@masked, regenerate returns full.
   - MainActivity onConnect: parses full string.
   - No "single 8-char token only" flow. (Server token itself is correct 8-char.)
2. **State loss on phone off / process kill / restart:**
   - Connection (url + token + registeredClientId) NOT persisted beyond "last_access_code" (which is the full old-style string).
   - On relaunch: isConnected=false, registeredClientId=null, must manually re-open dialog + re-enter.
   - No auto-re-register or resume of WS/FL session.
   - activeWeights + modelVersion restored (good), but session context lost.
3. **HTTP 409 on resume/reconnect/update submission:**
   - submit_update raises 409 if: not registered, round != current, base_model_version != server, or client_id already in client_updates.
   - Android: register ONLY happens inside onConnect (in dialog). On app restart: no re-register before WS or submit.
   - In onUpdateRequested (WS listener): uses `registeredClientId ?: error(...)` + submit with possibly stale token/round/version.
   - Reconnect WS does not trigger re-register.
   - Server registered_clients is in-memory only (lost on server restart too).
   - round_participants / client_updates cleared per round but no resume registration guard.
4. **F1 / metrics claim 0.98 not reproducible:**
   - No `output/bootstrap_eval_report.json` or `latest_eval.json` exist (ls confirmed).
   - /api/metrics/comparison: `evaluated=false` when files absent; dashboard shows "Awaiting server-side evaluation".
   - No real eval run in repo head. Claim in old reports/docs is false/misleading. (Server _run_evaluation calls prep_eval.py which would fail without data/TF.)
   - Actual metrics from coordinator are client-reported + server-eval only if run.
5. **Submit token bug in training flow:**
   - WS onUpdateRequested closure captures `accessCode` (mutable state var, may be stale) for submitUpdate header.
   - `registeredClientId` may be null post-restart.
   - currentServerUrl ok but overall.
6. **Dashboard / status still emits full "IP@TOKEN":** propagates old UX.
7. **No single source of truth for connection:** url/token split across prefs, state, parse.
8. **Other minor:**
   - Some in-memory only paths (e.g. smartAlbums restore only if photos non-empty).
   - WS listener submit doesn't re-validate registered after possible disconnect.
   - Server cleanup removes clients but no persisted client registry.
   - UDP discovery exists but not used in Android connect flow.

## What is MISLEADING or Falsely Claimed
- "F1 increase to 0.98" (old reports): no artifacts, no eval files, dashboard explicitly awaits eval. Not grounded.
- "Fully wired organize": code exists + buttons call, but real file system writes (Pictures/FGT) only happen if user taps; no auto on launch.
- "App loses no state": model + albums persist, but FL *session/connection* does not.
- "Simple 8-char" (recent commits): server side yes, but Android + dashboard UX still IP@ style.
- "No 409 on resume": not true in current head.

## Evidence Sources
- Code traces: MainActivity.kt (parse, dialog, WS listener, LaunchedEffect restores), FGTWebSocketClient.kt, fl_coordinator.py (submit checks), main.py (verify_token, 409, status), dashboard.js (access_code handling).
- Execution: server start + /status curl returned real token + model.
- Files: no eval jsons; config token="P54RCJHM"; schema match.
- No mocks in core paths (real trainer, serialize, coordinator).

**Bottom line:** Core FL protocol + serialization + gallery scan + some persistence is real and mostly correct. The blocking user reqs (token UX, device-off resume, 409, F1 evidence) have concrete defects in wiring/persistence/UI. Small targeted fixes only.

END TRUTH MAP
