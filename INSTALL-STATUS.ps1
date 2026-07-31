[CmdletBinding()]
param([string]$InstallRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio")
$ErrorActionPreference = 'Stop'
$StateRoot = Join-Path $InstallRoot 'install-state\stages'
if (-not (Test-Path $StateRoot)) {
  Write-Host 'No installer checkpoints exist yet.' -ForegroundColor Yellow
  exit 0
}
$rows = foreach ($path in Get-ChildItem $StateRoot -Filter '*.json' -File | Sort-Object Name) {
  try {
    $item = Get-Content $path.FullName -Raw | ConvertFrom-Json
    [pscustomobject]@{
      Stage = $item.name
      Status = $item.status
      Attempts = $item.attempts
      Updated = $item.updated_at
    }
  } catch {}
}
$rows | Format-Table -AutoSize
$failed = @($rows | Where-Object Status -eq 'failed')
$degraded = @($rows | Where-Object Status -eq 'degraded')
if ($failed.Count -gt 0) {
  Write-Host "`nFailed stage details:" -ForegroundColor Red
  foreach ($row in $failed) {
    $path = Join-Path $StateRoot ((($row.Stage.ToLowerInvariant() -replace '[^a-z0-9._-]+','-').Trim('-')) + '.json')
    $item = Get-Content $path -Raw | ConvertFrom-Json
    Write-Host "- $($item.name): $($item.error)" -ForegroundColor Red
  }
  Write-Host "`nRun CONTINUE-INSTALL.cmd. Verified completed stages will be skipped." -ForegroundColor Cyan
}

if ($degraded.Count -gt 0) {
  Write-Host "`nOptional degraded stages:" -ForegroundColor DarkYellow
  foreach ($row in $degraded) {
    $path = Join-Path $StateRoot ((($row.Stage.ToLowerInvariant() -replace '[^a-z0-9._-]+','-').Trim('-')) + '.json')
    $item = Get-Content $path -Raw | ConvertFrom-Json
    Write-Host "- $($item.name): $($item.error)" -ForegroundColor DarkYellow
  }
  Write-Host "`nCore installation can remain usable. Run RETRY-OPTIONAL.cmd only when you want to retry these stages." -ForegroundColor Cyan
}
