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
