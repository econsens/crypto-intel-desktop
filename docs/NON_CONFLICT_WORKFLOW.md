# Non-Conflict Git Workflow (Windows PowerShell)

Use this exact flow to avoid recurring PR conflicts.

## 1) Use the real PR branch name (never placeholders)

```powershell
git fetch origin --prune
git branch -a
git checkout <real-pr-branch-name>
$BRANCH = (git branch --show-current).Trim()
```

## 2) Rebase on latest main

```powershell
git fetch origin --prune
git rebase origin/main
```

## 3) If conflicts appear, resolve and continue

Check current state:

```powershell
git status
```

### Conflict in `app.py` (keep your branch version)

```powershell
git checkout --ours app.py
git add app.py
git rebase --continue
```

### Conflict in `docs/CHANGELOG.md`

Quick path:

```powershell
git checkout --ours docs/CHANGELOG.md
git add docs/CHANGELOG.md
git rebase --continue
```

Manual path (only if you need both sides): remove conflict markers, keep valid bullets, then:

```powershell
git add docs/CHANGELOG.md
git rebase --continue
```

Repeat conflict resolution until rebase completes.

## 4) Understand status messages

- `Successfully rebased and updated refs/heads/...` → rebase is done; push next.
- `fatal: no rebase in progress` → no active rebase; push or start rebase.
- `interactive rebase in progress` in `git status` → keep resolving and running `git rebase --continue`.

## 5) Push safely after rebase

```powershell
git push --force-with-lease origin $BRANCH
```

## 6) Final check

```powershell
git status
```

Expected: `nothing to commit, working tree clean`.

## Emergency reset

```powershell
git rebase --abort
git fetch origin --prune
git rebase origin/main
```
