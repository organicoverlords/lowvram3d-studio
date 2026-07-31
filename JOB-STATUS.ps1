[CmdletBinding()]
param([string]$InstallRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio")
$ErrorActionPreference = 'Stop'
$JobsRoot = Join-Path $InstallRoot 'jobs'
if (-not (Test-Path $JobsRoot)) {
  Write-Host 'No asset jobs exist yet.' -ForegroundColor Yellow
  exit 0
}
$rows = foreach ($receiptPath in Get-ChildItem $JobsRoot -Filter job_receipt.json -Recurse -File -ErrorAction SilentlyContinue) {
  try {
    $receipt = Get-Content $receiptPath.FullName -Raw | ConvertFrom-Json
    $lastStage = @($receipt.stages | Select-Object -Last 1)[0]
    [pscustomobject]@{
      Job = $receipt.job_id
      Operation = $receipt.operation
      Status = $receipt.status
      LastStage = if ($lastStage) { "$($lastStage.stage):$($lastStage.status)" } else { '' }
      Started = $receipt.started_at
      Receipt = $receiptPath.FullName
    }
  } catch {}
}
$rows = @($rows | Sort-Object Started -Descending)
if ($rows.Count -eq 0) {
  Write-Host 'No readable asset-job receipts exist yet.' -ForegroundColor Yellow
  exit 0
}
$rows | Select-Object -First 20 Job,Operation,Status,LastStage | Format-Table -AutoSize
$failed = @($rows | Where-Object Status -eq 'failed')
if ($failed.Count -gt 0) {
  Write-Host "`nLatest failed resumable job: $($failed[0].Job)" -ForegroundColor Cyan
  Write-Host 'Use Resume Last 3D Job or rerun the same card in 3D Gen Studio.' -ForegroundColor Cyan
}
