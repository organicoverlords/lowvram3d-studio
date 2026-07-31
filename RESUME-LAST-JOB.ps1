[CmdletBinding()]
param(
  [string]$InstallRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio",
  [string]$WorkerUrl = 'http://127.0.0.1:8400'
)
$ErrorActionPreference = 'Stop'
$AppRoot = Join-Path $InstallRoot 'app'
$JobsRoot = Join-Path $InstallRoot 'jobs'
if (-not (Test-Path $JobsRoot)) { throw 'No asset jobs exist yet.' }

function Test-Worker {
  try {
    $health = Invoke-RestMethod -Uri "$WorkerUrl/health" -TimeoutSec 3
    return ([string]$health.status -eq 'ok')
  } catch { return $false }
}

if (-not (Test-Worker)) {
  Write-Host 'Starting local 3D services...' -ForegroundColor Cyan
  & (Join-Path $AppRoot 'START-STUDIO.ps1') -NoBrowser
  if ($LASTEXITCODE -ne 0 -or -not (Test-Worker)) { throw 'LowVRAM worker did not become ready.' }
}

$failed = foreach ($receiptPath in Get-ChildItem $JobsRoot -Filter job_receipt.json -Recurse -File -ErrorAction SilentlyContinue) {
  try {
    $receipt = Get-Content $receiptPath.FullName -Raw | ConvertFrom-Json
    if ($receipt.status -eq 'failed' -and $receipt.operation -in @('full','postprocess')) {
      [pscustomobject]@{ Receipt=$receipt; Path=$receiptPath.FullName; Started=[long]$receipt.started_at }
    }
  } catch {}
}
$target = @($failed | Sort-Object Started -Descending | Select-Object -First 1)[0]
if (-not $target) { throw 'No failed full/postprocess job is available to resume.' }
$jobId = [string]$target.Receipt.job_id
Write-Host "Resuming job $jobId from its first failed or invalid stage..." -ForegroundColor Cyan
$downloadDir = Join-Path $InstallRoot 'resumed-output'
New-Item -ItemType Directory -Force -Path $downloadDir | Out-Null
$output = Join-Path $downloadDir "$jobId.glb"
try {
  Invoke-WebRequest -UseBasicParsing -Method Post -Uri "$WorkerUrl/v1/jobs/$jobId/resume" -OutFile $output -TimeoutSec 86400
} catch {
  Write-Host "Resume failed again. The same job remains checkpointed: $jobId" -ForegroundColor Yellow
  Write-Host "Receipt: $($target.Path)" -ForegroundColor Yellow
  throw
}
if (-not (Test-Path $output) -or (Get-Item $output).Length -eq 0) { throw 'Resume endpoint returned no GLB.' }
Write-Host "Resume completed: $output" -ForegroundColor Green
Write-Host "Original job outputs and proof remain under: $(Split-Path -Parent (Split-Path -Parent $target.Path))" -ForegroundColor Green
