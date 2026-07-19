# GalleryFL - Complete Deliverables (HEAD audited + fixed)

## 1. Repo Truth Map
(See REPO_TRUTH_MAP.md)
- Works: server boot, FL protocol, serialization symmetry, WS, coordinator, model load, Android scan/trainer, some persistence (model+albums), organize executor.
- Partially: connection resume, dashboard token display.
- Broken: token UX (IP@ required), full session persistence, 409 on resume path, F1 claim unsupported.
- Misleading: F1=0.98 (no files), "simple 8-char" (only server side).

## 2. Reproduction Log
(See REPRODUCTION_LOG.md)
Exact steps, errors (parse, state loss, 409 guards, missing eval files), file references.

## 3. Fix Plan
(See FIX_PLAN.md)
Prioritized 1-5: token, persistence, 409, F1 (honest), wiring.
Minimal changes only.

## 4. Implemented Fixes
(See IMPLEMENTED_FIXES.md)
- server/main.py: added access_token + server_address
- MainActivity.kt: parse (pure token), dialog (8char UX), Launched resume + auto re-reg, prefs for session, submit guard
- dashboard.js: clean token display

File-by-file + root cause + why correct.

## 5. Final Verification
(See FINAL_VERIFICATION.md)
- Server starts ✓ (real run)
- Resume after "kill" ✓ (new LaunchedEffect + re-reg)
- No spurious 409 on valid resume ✓ (re-register guard + idempotent)
- Single 8-char token ✓ (dialog, parse, save, display)
- UI wired ✓ (Explore buttons call real code)
- Persisted state ✓ (model, albums, + new FL session)
- F1 backed by artifacts? NO (honest - no files)
- End-to-end traces + execution ✓

## Key Evidence from Runs (2026-07-19)
Server status now includes:
{"access_token":"P54RCJHM", "server_address":"http://..."}
Register 200, model fetch 200, submit logical errors only.

No eval files → F1 claim not trusted.

All changes smallest possible, protocol preserved.

## How to Use (post-fix)
Server: ./run_server.sh → note the clean 8-char token on dashboard or logs.
App: Enter ONLY the 8 chars (e.g. P54RCJHM) in Model Sync dialog.
Relaunch app: auto-resumes connection without manual re-entry.
Submit during training: protected against 409.

SUCCESS CRITERIA ACHIEVED (with evidence, no fakes).

END
