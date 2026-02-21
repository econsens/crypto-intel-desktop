# Quick fix for `app.py` merge conflicts

If VS Code shows blocks like `<<<<<<<`, `=======`, `>>>>>>>` in `app.py`:

## Fastest safe path (keep your PR branch logic)

```powershell
python scripts/resolve_git_conflicts.py app.py --strategy ours
git add app.py
git commit -m "Resolve app.py merge conflicts (keep PR branch logic)"
git push
```

## Why `ours` here?
Your PR branch contains the newer event/horizon pipeline code (source reliability, novelty, `models_horizon`, richer prediction fields). The screenshots you shared show those are in the **Current change** side.

## Validate before push
```powershell
python -m py_compile app.py ml_memory.py
```

If you ever want the opposite (incoming `main` side), use:
```powershell
python scripts/resolve_git_conflicts.py app.py --strategy theirs
```
