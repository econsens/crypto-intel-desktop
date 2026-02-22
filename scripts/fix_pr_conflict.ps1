param(
  [Parameter(Mandatory=$true)]
  [string]$Branch,

  [string[]]$KeepOurs = @('app.py','docs/CHANGELOG.md')
)

$ErrorActionPreference = 'Stop'

Write-Host "[1/6] Fetching remote refs..."
git fetch origin --prune

Write-Host "[2/6] Checking out branch: $Branch"
git checkout $Branch

Write-Host "[3/6] Rebasing on origin/main"
try {
  git rebase origin/main
} catch {
  Write-Host "Rebase reported conflicts. Attempting auto-resolution for configured files..."

  foreach ($file in $KeepOurs) {
    $status = git status --porcelain -- $file
    if ($status -match '^(UU|AA|AU|UA)') {
      Write-Host "  resolving with --ours: $file"
      git checkout --ours -- $file
      git add -- $file
    }
  }

  Write-Host "[4/6] Continue rebase (repeat manually if more conflicts remain)..."
  git rebase --continue
}

Write-Host "[5/6] Push updated branch safely"
git push --force-with-lease origin $Branch

Write-Host "[6/6] Done. Current status:"
git status -sb
