param(
  [Parameter(Mandatory=$true)]
  [string]$Branch,

  [switch]$KeepOursAppPy
)

$ErrorActionPreference = 'Stop'

Write-Host "==> Fetching remote branches"
git fetch origin --prune

Write-Host "==> Checking out PR branch: $Branch"
$exists = git branch --list $Branch
if (-not $exists) {
  git checkout -b $Branch origin/$Branch
} else {
  git checkout $Branch
}

Write-Host "==> Pull latest branch state"
git pull --ff-only origin $Branch

Write-Host "==> Merging origin/main into $Branch"
$mergeOk = $true
try {
  git merge origin/main
} catch {
  $mergeOk = $false
}

if (-not $mergeOk) {
  if ($KeepOursAppPy) {
    Write-Host "==> Merge conflict detected. Keeping branch version of app.py"
    git checkout --ours app.py
    git add app.py
    git commit -m "Resolve conflict in app.py"
  } else {
    Write-Host "Merge conflict detected. Resolve files manually, then run:"
    Write-Host "  git add <files>"
    Write-Host "  git commit -m 'Resolve merge conflicts'"
    exit 1
  }
}

Write-Host "==> Pushing branch"
git push -u origin $Branch

Write-Host "Done. Refresh GitHub PR page."
