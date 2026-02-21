# Non-Conflict Git Workflow (Windows PowerShell)

This workflow avoids the two most common mistakes we saw:
- typing placeholder names like `YOUR_BRANCH_NAME`;
- getting stuck in an in-progress rebase with `docs/CHANGELOG.md` conflicts.

## 0) Always run one command per line

Do **not** paste two commands on one line.

✅ Good:

```powershell
git checkout my-branch
git rebase origin/main
```

❌ Bad:

```powershell
git checkout my-branchgit rebase origin/main
```

## 1) Set your branch name once (no placeholders)

```powershell
$BRANCH = (git branch --show-current).Trim()
$BRANCH
```

If this prints empty, stop and run `git checkout <your-branch>` first.

## 2) Sync remote refs

```powershell
git fetch origin --prune
```

## 3) Rebase your current branch on latest `main`

```powershell
git rebase origin/main
```

## 4) If rebase conflicts, resolve and continue

Check which files are conflicted:

```powershell
git status
```

### 4a) Conflict in `app.py` (keep your branch copy)

```powershell
git checkout --ours app.py
git add app.py
git rebase --continue
```

### 4b) Conflict in `docs/CHANGELOG.md`

Open the file, remove conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`), keep both valid changelog bullets, then:

```powershell
git add docs/CHANGELOG.md
git rebase --continue
```

Repeat until Git says rebase is complete.

## 5) Push rebased branch safely

```powershell
git push --force-with-lease origin $BRANCH
```

## 6) Final check

```powershell
git status
```

You want: `nothing to commit, working tree clean`.

---

## Emergency reset if you get stuck

```powershell
git rebase --abort
git fetch origin --prune
git rebase origin/main
```
