# Non-Conflict Git Workflow (Windows PowerShell)

Use this exact flow every time before opening/updating a PR.

## 1) Start from a clean repo

```powershell
git status
```

If you have local edits, either commit them or stash them first.

## 2) Update local references

```powershell
git fetch origin --prune
```

## 3) Switch to your PR branch

```powershell
git checkout YOUR_BRANCH_NAME
```

> Do not use `< >` in commands.

## 4) Rebase your branch on latest `main`

```powershell
git rebase origin/main
```

If conflict appears in `app.py` and you want to keep your branch version:

```powershell
git checkout --ours app.py
git add app.py
git rebase --continue
```

Repeat those 3 lines until rebase completes.

## 5) Push safely after rebase

```powershell
git push --force-with-lease origin YOUR_BRANCH_NAME
```

## 6) Verify branch is clean

```powershell
git status
```

You should see: `nothing to commit, working tree clean`.

---

## Daily short version (copy/paste)

```powershell
git fetch origin --prune
git checkout YOUR_BRANCH_NAME
git rebase origin/main
git push --force-with-lease origin YOUR_BRANCH_NAME
```

---

## Optional: if rebase gets messy

Abort and restart cleanly:

```powershell
git rebase --abort
git fetch origin --prune
git rebase origin/main
```
