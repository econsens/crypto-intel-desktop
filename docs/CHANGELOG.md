# Changelog

## 2026-02-21
- Removed Git conflict-recovery helper docs and scripts from the repository to keep this project focused on runtime product behavior.
- Kept the event/horizon prediction pipeline and memory/runtime updates intact.
- Added `docs/NON_CONFLICT_WORKFLOW.md` with a simple rebase-first PowerShell workflow to prevent recurring PR conflicts.
- Clarified `docs/NON_CONFLICT_WORKFLOW.md` to explain `no rebase in progress`/`rebase complete` states and enforce selecting the exact PR branch before rebasing.
- Simplified `docs/NON_CONFLICT_WORKFLOW.md` to a short canonical sequence to reduce repeated merge conflicts in overlapping line ranges.
