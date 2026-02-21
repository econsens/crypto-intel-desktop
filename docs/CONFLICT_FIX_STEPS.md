# GitHub PR Conflict Fix (PowerShell-safe)

Use these exact commands in **PowerShell** (no `< >` placeholders).

## 1) See branches
```powershell
git fetch origin --prune
git branch -a
```

## 2) Switch to the exact PR branch
Example PR branch: `codex/find-information-on-crypto-intel-329fiu`
```powershell
git checkout -b codex/find-information-on-crypto-intel-329fiu origin/codex/find-information-on-crypto-intel-329fiu
```
If it already exists locally:
```powershell
git checkout codex/find-information-on-crypto-intel-329fiu
```

## 3) Merge latest main
```powershell
git merge origin/main
```

If conflict appears only in `app.py` and you want to keep your PR-branch copy:
```powershell
git checkout --ours app.py
git add app.py
git commit -m "Resolve conflict in app.py"
```

## 4) Push
```powershell
git push -u origin codex/find-information-on-crypto-intel-329fiu
```

## 5) Refresh GitHub PR page
The conflict banner should be gone.

---

## Optional one-command helper
You can also run:
```powershell
pwsh -File scripts/sync-pr-branch.ps1 -Branch codex/find-information-on-crypto-intel-329fiu -KeepOursAppPy
```
