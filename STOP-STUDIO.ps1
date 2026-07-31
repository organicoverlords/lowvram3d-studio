$ErrorActionPreference = 'Continue'
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallRoot = Split-Path -Parent $AppRoot
$PidDir = Join-Path $InstallRoot 'pids'

function Read-PidRecord([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  $raw = (Get-Content $Path -Raw -ErrorAction SilentlyContinue).Trim()
  if (-not $raw) { return $null }
  try { return $raw | ConvertFrom-Json }
  catch {
    $value = 0
    if ([int]::TryParse($raw,[ref]$value)) { return [pscustomobject]@{ pid=$value; legacy=$true } }
    return $null
  }
}

function Test-RecordMatchesProcess($Record,[System.Diagnostics.Process]$Process) {
  if (-not $Record -or -not $Process) { return $false }
  if (-not ($Record.PSObject.Properties.Name -contains 'process_start_time')) { return $true }
  try {
    $recorded = [DateTime]::Parse([string]$Record.process_start_time).ToUniversalTime()
    return ([Math]::Abs(($Process.StartTime.ToUniversalTime() - $recorded).TotalSeconds) -lt 5)
  } catch { return $false }
}

foreach ($name in @('3d-gen-studio','lowvram-worker','meshtools','comfyui')) {
  $file = Join-Path $PidDir "$name.pid"
  $record = Read-PidRecord $file
  if ($record -and [int]$record.pid -gt 0) {
    $process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if ($process -and (Test-RecordMatchesProcess $record $process)) {
      & taskkill.exe /PID ([int]$record.pid) /T /F | Out-Null
    }
  }
  Remove-Item $file -Force -ErrorAction SilentlyContinue
}
Write-Host 'LowVRAM 3D Studio services stopped.'
