# Implemented Fixes - File-by-file Report

## 1. server/main.py
**Changes:**
- `/api/training/status` now returns `"access_token": config.server_token, "server_address": f"http://{lan}:{port}"` (in addition to legacy access_code).
- `/api/training/regenerate-token` also returns the new fields.

**Root cause:** Dashboard and clients expected only legacy "IP:PORT@TOKEN". Single source of truth for clean token was missing.
**Why correct:** Enables simple 8-char UX without breaking old paths. Minimal additive change.

## 2. android/.../MainActivity.kt
**Changes (multiple targeted edits):**

a. **parseConnectionDetails** (simplified):
   - Now supports *pure 8-char token* (new UX) OR legacy "IP@TOKEN".
   - Pure token case uses fallbackUrl + token.

b. **FLSyncDialog**:
   - Label/placeholder: "Access Token (8 chars)", example "P54RCJHM".
   - Validation: accepts pure token or extracts after @.
   - Generate button now produces pure 8-char.
   - Error messages updated for single-token UX.
   - Description text updated.

c. **Persistence & resume (new LaunchedEffect + state init)**:
   - Added `savedServerUrl`, `savedToken`, `savedRegisteredId` from prefs.
   - `currentServerUrl`, `accessCode`, `registeredClientId` init from persisted.
   - New `LaunchedEffect(Unit)`: on launch, if persisted url+token → re-register (idempotent), fetch model, save state, WS connect, set isConnected.
   - Saves `last_server_url`, `last_access_token`, `last_registered_client_id` on connect + during resume.

d. **onConnect lambda**:
   - Extracts token-only, saves url + token (not full legacy string) to prefs.
   - Uses token for register/WS.
   - Persists registered id.

e. **WS onUpdateRequested submit path**:
   - Before submit: re-register using live persisted token/clientId (prevents "not registered" → 409).
   - Uses latest `last_access_token` for submit header.
   - Falls back safely.

**Root cause:** 
- Token UX was hardcoded to composite string.
- No persisted connection/session state → state loss + no re-reg on restart.
- submit used closure-captured (potentially stale) vars + no re-register guard → 409 on resume.
**Why correct:** Smallest correct fix. Uses existing prefs + register idempotency. Re-register before submit is defensive but cheap. Preserves all FL paths.

## 3. server/dashboard/dashboard.js
**Changes:**
- `maskCode`: prefers clean token display.
- `renderCode`: extracts token if legacy format present.
- Status refresh: prefers `server_address` / `access_token` from API.
- Copy uses full or token as appropriate.

**Root cause:** UI was showing/requiring composite access_code.
**Why correct:** Users now see simple token prominently.

## Summary of Minimalism
- No rewrites.
- No mocks.
- 8-char token is now the primary UX input.
- Persistence + auto-re-register directly addresses device-off + 409.
- All core FL, serialize, WS, coordinator, Room paths untouched.

## Files touched (diff summary)
- server/main.py: +3 lines (new fields)
- android/MainActivity.kt: ~80 lines changed (parse, dialog, 2 Launched, connect, submit guard)
- server/dashboard/dashboard.js: ~15 lines (display logic)

END FIXES REPORT
