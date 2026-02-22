# PR Conflict One-Shot Fix (PowerShell)

If GitHub says your PR has conflicts, run this from your repo root in **PowerShell**.

## Option A (recommended): one command script

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_pr_conflict.ps1 -Branch "<your-pr-branch>"
```

Example:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\fix_pr_conflict.ps1 -Branch "codex/find-information-on-crypto-intel-x189fq"
```

## Option B: manual exact commands

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

## Notes

- Do **not** type placeholders like `YOUR_PR_BRANCH` literally.
- Do **not** add `< >` around branch names in PowerShell.
- `fatal: no rebase in progress` means there is no active rebase step to continue.
