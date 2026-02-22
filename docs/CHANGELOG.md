# Changelog

## 2026-02-21
- Removed Git conflict-recovery helper docs and scripts from the repository to keep this project focused on runtime product behavior.
- Kept the event/horizon prediction pipeline and memory/runtime updates intact.
- Added `docs/NON_CONFLICT_WORKFLOW.md` with a simple rebase-first PowerShell workflow to prevent recurring PR conflicts.
- Clarified `docs/NON_CONFLICT_WORKFLOW.md` to explain `no rebase in progress`/`rebase complete` states and enforce selecting the exact PR branch before rebasing.
- Simplified `docs/NON_CONFLICT_WORKFLOW.md` to a short canonical sequence to reduce repeated merge conflicts in overlapping line ranges.
- Added `scripts/fix_pr_conflict.ps1` and `docs/PR_CONFLICT_ONE_SHOT.md` for one-command PR conflict resolution in PowerShell.
- Added troubleshooting for missing `scripts/fix_pr_conflict.ps1` in `docs/PR_CONFLICT_ONE_SHOT.md` with a manual fallback flow.
- Dockerfile now copies `assets/` into the image to prevent runtime 500 errors when the dashboard template is missing.
- Added UI fallback in `app.py` when `assets/dashboard_template.html` is missing, with a clear rebuild hint.
- Updated conflict docs to require `git pull --ff-only` when a branch is behind before rebasing/pushing.
- Added `template_exists`/`template_path` to `/debug/runtime` and new `/debug/template` endpoint to diagnose main-page template issues quickly.
- Expanded `/health` payload with `version`, `model_version`, and `template_exists` to quickly detect stale containers.
- Added `docs/FIX_FROM_ZERO_WINDOWS.md` with beginner-safe copy/paste steps for conflict resolution + Docker redeploy.
- Added `/debug/routes` to list active FastAPI paths, making stale-container route mismatches easy to detect.
- Added `/debug/signature` and included signature/route hints in `/health` to verify running container code version.
- Updated Windows recovery guide to run from `main` after PR merges and check `/debug/signature`.
- Startup now captures non-fatal initialization errors instead of crashing the whole app process.
- Added `/debug/startup` and included `startup_errors` in `/health` for immediate crash-cause visibility.
- Switched SQLite connections to WAL + busy timeout via `db_connect()` to reduce `database is locked` errors in concurrent loops.
- Updated Windows diagnostics commands to use `curl.exe -s` to avoid PowerShell Invoke-WebRequest script warnings.
