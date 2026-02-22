# PR Conflict One-Shot Fix (PowerShell)

If GitHub says your PR has conflicts, run this from your repo root in **PowerShell**.

## Option A (recommended): one-command script

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_pr_conflict.ps1 -Branch "<your-pr-branch>"
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_pr_conflict.ps1 -Branch "codex/find-information-on-crypto-intel-x189fq"
```

## Option B: manual exact commands (works even if script file is missing)


## Important: if branch is "behind" remote, update first

If `git status` says your local branch is behind origin, run:

```powershell
git pull --ff-only
```

Then continue with rebase/resolve steps. This prevents accidentally force-pushing older history back to GitHub.

```powershell
git fetch origin --prune
git checkout <your-pr-branch>
git rebase origin/main
```

If conflict appears in `app.py` or `docs/CHANGELOG.md`:

```powershell
git checkout --ours app.py docs/CHANGELOG.md
git add app.py docs/CHANGELOG.md
git rebase --continue
```

Repeat `checkout/add/rebase --continue` until rebase finishes.

Then push:

```powershell
git push --force-with-lease origin <your-pr-branch>
```

## If you get: `-File ...fix_pr_conflict.ps1 does not exist`

You are on a branch that does not contain `scripts/fix_pr_conflict.ps1` yet.

Run this immediately:

```powershell
git fetch origin --prune
git branch -a
```

Then either:

1. Switch to the branch that includes the script and run Option A, **or**
2. Stay on your current branch and run Option B manual commands above.

You can verify whether the script exists in your current checkout with:

```powershell
Test-Path .\scripts\fix_pr_conflict.ps1
```

- `True` = script is present.
- `False` = use Option B or checkout the branch that includes the script.

## Notes

- Do **not** type placeholders like `YOUR_PR_BRANCH` literally.
- Do **not** add `< >` around branch names in PowerShell.
- `fatal: no rebase in progress` means there is no active rebase step to continue.
