# Non-Conflict Git Workflow (Windows PowerShell)

This workflow avoids the most common mistakes:
- typing placeholder names like `YOUR_BRANCH_NAME`;
- typing two commands on one line;
- running `git rebase --continue` when no rebase is active;
- trying to push a different branch than the one the PR uses.

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

## 1) Identify the exact PR branch first

```powershell
git fetch origin --prune
git branch -a
```

Copy the real PR branch name from GitHub (for example `codex/find-information-on-crypto-intel-x189fq`) and check it out:

```powershell
git checkout codex/find-information-on-crypto-intel-x189fq
```

## 2) Set your current branch variable (no placeholders)

```powershell
$BRANCH = (git branch --show-current).Trim()
$BRANCH
```

If this prints empty, stop and run `git checkout <real-branch-name>` first.

## 3) Rebase on latest `main`

```powershell
git fetch origin --prune
git rebase origin/main
```

## 4) If rebase conflicts, resolve and continue

Check conflict files:

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

Fast path (keep current branch version):

```powershell
git checkout --ours docs/CHANGELOG.md
git add docs/CHANGELOG.md
git rebase --continue
```

Manual path (if you need both sides):
1. Open `docs/CHANGELOG.md`.
2. Remove `<<<<<<<`, `=======`, `>>>>>>>` markers.
3. Keep valid bullets.
4. Save file, then run:

```powershell
git add docs/CHANGELOG.md
git rebase --continue
```

Repeat until Git says rebase is complete.

## 5) Important status meanings

### A) `fatal: no rebase in progress`
This is **not** a new conflict. It means you already finished (or were never in) a rebase. Next step is push.

### B) `Successfully rebased and updated refs/heads/...`
Rebase is done. Do **not** run conflict commands after this line. Just push.

### C) `interactive rebase in progress` in `git status`
Rebase is still active. Keep resolving conflicts + `git rebase --continue`.

## 6) Push rebased branch safely

```powershell
git push --force-with-lease origin $BRANCH
```

## 7) Final check

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
