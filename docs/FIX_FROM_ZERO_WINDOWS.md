# Fix From Zero (Windows, beginner-safe)

Use this if:
- browser shows `Internal Server Error`, or
- `http://127.0.0.1:8000/debug/template` says `{"detail":"Not Found"}`.

`Not Found` for `/debug/template` means your running container is still old code.

## Step 1) Open PowerShell in your repo folder

You should be inside your project folder (example: `C:\crypto-intel-mini`).

## Step 2) Update your branch to latest code

Replace branch name with the PR branch you are working on.

```powershell
git fetch origin --prune
git checkout <your-pr-branch>
git pull --ff-only
git rebase origin/main
git push --force-with-lease origin <your-pr-branch>
```

If rebase conflict appears:

```powershell
git checkout --ours app.py docs/CHANGELOG.md
git add app.py docs/CHANGELOG.md
git rebase --continue
```

Repeat until rebase finishes, then run push again.

## Step 3) Rebuild Docker with no cache (important)

```powershell
docker stop crypto-mini
docker rm crypto-mini
docker build --no-cache -t crypto-mini:latest .
docker run -d --name crypto-mini -p 127.0.0.1:8000:8000 -v C:/crypto-intel-data:/data crypto-mini:latest
```

## Step 4) Verify diagnostics

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/debug/runtime
curl http://127.0.0.1:8000/debug/template
```

Expected:
- `/health` includes `version` and `template_exists`.
- `/debug/template` exists and returns JSON (not `Not Found`).

## Step 5) Open browser and hard refresh

Open:
- `http://127.0.0.1:8000/`

Then hard refresh:
- `Ctrl + F5`

## If still broken

Run this and copy output:

```powershell
docker logs --tail 120 crypto-mini
```

Share that log output so we can pinpoint the exact error.
