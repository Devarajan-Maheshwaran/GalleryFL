# Fix Plan - GalleryFL (minimal, evidence-based, prioritized)

Prioritized by severity (user requirements + blocking correctness):

## Priority 1: Token UX (Critical User Req - "one single 8-char alphanumeric token only")
- Root: parseConnectionDetails + dialog enforce "IP:PORT@TOKEN"; dashboard emits full string.
- Fix: Decouple URL and token. Dialog gets Server URL (prefilled) + pure 8-char Access Code. 
  Server status returns separate fields. Update all parsing/display.
- Files: MainActivity.kt, FGTApiService (minor), server/main.py, server/dashboard/dashboard.js, server/config.py (no change).
- Smallest: keep url separate, token only 8char validated.

## Priority 2: State loss on shutdown/restart (Critical)
- Root: only "last_access_code" (old format) + model weights persisted. No url/token/clientId.
- Fix: Persist serverUrl, accessToken, registeredClientId in SharedPrefs. Restore + auto-reconnect logic in LaunchedEffect.
- Add explicit connection state persistence.

## Priority 3: HTTP 409 on resume/reconnect (Critical)
- Root: register only in manual onConnect; WS update path assumes registeredClientId + uses potentially stale accessCode; no re-register on resume.
- Fix: 
  - Persisted state triggers auto-register on launch.
  - Before any submit (in onUpdateRequested): ensure registered (re-register using persisted if needed).
  - Pass latest token explicitly to submit.
  - Server register is idempotent (already).
  - Add re-register helper.

## Priority 4: F1/metrics claims
- Root: no output/*.eval.json files; claims only in old docs.
- Fix: Do not touch claims. In code, /comparison already honest. No reproduction possible without TF+data (leave as-is, document in truth map). Do not fake runs.

## Priority 5: Other logical/UI wiring
- Ensure ExploreScreen Organize/Undo actually wired (they are).
- Fix potential stale var capture in WS listener submit.
- Single source: use prefs for connection.
- Add small resilience: re-register before WS send if needed.
- No overengineer.

## Order of implementation
1. Server status + dashboard (clean token display) - minimal.
2. Android dialog + parse + connect (simplify to URL + 8char token).
3. Persistence helpers + Launched restore + auto connect.
4. Submit/WS path hardening against 409 (re-register guard).
5. Verify with simulated runs.
6. Update truth map + final verification section.

## Verification criteria (to be run after)
- Server starts, status shows clean token.
- Dialog accepts 8-char only (no forced @IP).
- Connect uses just token.
- After "kill process" (sim), relaunch restores connection + re-registers without 409.
- Model weights + albums + session resume.
- Real submit path succeeds.
- No dead routes.

Minimal diffs only. Preserve FL protocol.
END PLAN
