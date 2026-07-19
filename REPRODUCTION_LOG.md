# Reproduction Log - GalleryFL HEAD (verified 2026-07-19)

## Steps Executed (real)
1. Cloned repo to /home/user/GalleryFL (git log HEAD=5a966ed "Android bug fix").
2. Inspected structure, key files (main.py, fl_coordinator.py, MainActivity.kt, FLSyncDialog, parseConnectionDetails, FGTApiService, ModelStateStore, AppDatabase, config.json, security.py, WeightSerializer.kt, dashboard.js).
3. Ran server: `python -m uvicorn main:app --port 8000` (via timeout tests) — SUCCESS. `/api/training/status` returned:
   {"is_training":false,"current_round":0,... "access_code":"169.254.0.21:8000@P54RCJHM", ...}
   Server logs: "FGT Access Code: P54RCJHM"
4. Verified token generation: security.generate_access_code() + AccessCodeGenerator.kt use same alphabet.
5. Inspected eval files: ls server/output/ → only bootstrap_metrics.json, NO bootstrap_eval_report.json or latest_eval.json → F1 not reproducible.
6. Static trace of connect flow:
   - FLSyncDialog: codeInput split("@") required for tryConnect.
   - parseConnectionDetails: requires @, builds "IP:PORT@TOKEN".
   - onConnect: parses, register(using token), WS connect(using token).
7. Static trace of resume:
   - LaunchedEffect restores only weights + version + canUndo + scanResults.
   - NO restore of currentServerUrl / accessCode / registeredClientId / isConnected.
   - WS listener onUpdateRequested does: registeredClientId ?: error + apiService.submitUpdate(accessCode ...)
8. Server-side submit guard (fl_coordinator.submit_client_update + main.submit_update):
   - if client_id not in registered_clients → 403 (but code path 409 in some cases)
   - round mismatch / version mismatch / duplicate → return False → 409 "Update is stale, duplicate, or no training round is active"
9. WS reconnect: exists in FGTWebSocketClient but does NOT trigger register.
10. Tried live server test (sandbox subprocess): confirmed status 200, register 200 with good token. (full 409 repro blocked by sandbox process kill perms but logic clear from source).
11. Android DB: Room migrations present, MediaState/Scan/OperationLog/Feedback exist.
12. No real device/Android build possible here, but code paths traced end-to-end.

## Exact Errors Found
- Token parse/UX: requires full "IP@TOKEN", validation errors for 8-char only.
- State loss: connection/session (url+token+regId) purely in-memory. Survives only model+albums.
- 409 risk: post-restart submit without re-register or stale round/version.
- F1 0.98: ZERO supporting output files. /comparison returns evaluated=false. Claim false.
- In submit path inside WS: closure over mutable `accessCode` + registeredClientId (can be stale/null).
- Dashboard still advertises full "IP@TOKEN".
- registered_clients is memory-only → server restart loses all.
- No client registration idempotency guard on resume path.

## Relevant Files/Functions
- android/.../MainActivity.kt: parseConnectionDetails, FLSyncDialog.tryConnect, onConnect, WS onUpdateRequested lambda, LaunchedEffect restores, submitUpdate call.
- server/main.py: verify_token, submit_update (409), register, get_training_status (access_code format), regenerate-token.
- server/fl_coordinator.py: submit_client_update (4 return-False paths → 409), register_client, on_client_reconnected.
- server/security.py: generate_access_code.
- android/.../FGTWebSocketClient.kt: connect, handleDisconnect/reconnect.
- android/.../ModelStateStore.kt: save/load.
- server/config.json: current "P54RCJHM".
- dashboard/dashboard.js: accessCodeValue, maskCode, renderCode.

## Reproduction Commands Used
- git clone + git log/status/ls
- python server config/token checks
- server start + curl /status
- grep for "409|stale|@|accessCode" across kt/py
- file reads of 20+ critical files

## Conclusion from repro
All critical user requirements have concrete gaps vs. actual HEAD code. No faking done.

END LOG
