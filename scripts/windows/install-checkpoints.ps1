Set-StrictMode -Version Latest

function Write-JsonUtf8NoBom {
  param(
    [Parameter(Mandatory=$true)][string]$Path,
    [Parameter(Mandatory=$true)]$Value,
    [int]$Depth = 12
  )
  $json = $Value | ConvertTo-Json -Depth $Depth
  [System.IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding($false)))
}

function ConvertTo-CheckpointName {
  param([Parameter(Mandatory=$true)][string]$Name)
  return (($Name.ToLowerInvariant() -replace '[^a-z0-9._-]+','-').Trim('-'))
}

function Get-StageCheckpointPath {
  param(
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [Parameter(Mandatory=$true)][string]$Name
  )
  return Join-Path $StateRoot ((ConvertTo-CheckpointName $Name) + '.json')
}

function Read-StageCheckpoint {
  param(
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [Parameter(Mandatory=$true)][string]$Name
  )
  $path = Get-StageCheckpointPath -StateRoot $StateRoot -Name $Name
  if (-not (Test-Path $path)) { return $null }
  try { return Get-Content $path -Raw | ConvertFrom-Json } catch { return $null }
}

function Write-StageCheckpoint {
  param(
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Fingerprint,
    [Parameter(Mandatory=$true)][string]$Status,
    [string]$ErrorText = '',
    [int]$Attempts = 0,
    [string]$StartedAt = '',
    [hashtable]$Details = @{}
  )
  New-Item -ItemType Directory -Force -Path $StateRoot | Out-Null
  $path = Get-StageCheckpointPath -StateRoot $StateRoot -Name $Name
  $payload = [ordered]@{
    name = $Name
    fingerprint = $Fingerprint
    status = $Status
    attempts = $Attempts
    started_at = $StartedAt
    updated_at = (Get-Date -Format o)
    completed_at = if ($Status -eq 'completed') { Get-Date -Format o } else { '' }
    error = $ErrorText
    details = $Details
  }
  $tmp = "$path.tmp"
  Write-JsonUtf8NoBom -Path $tmp -Value $payload -Depth 12
  Move-Item -Force $tmp $path
}

function Test-Readiness {
  param([Parameter(Mandatory=$true)][scriptblock]$Test)
  try { return [bool](& $Test) } catch { return $false }
}

function Invoke-InstallStage {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$Fingerprint,
    [Parameter(Mandatory=$true)][scriptblock]$Test,
    [Parameter(Mandatory=$true)][scriptblock]$Action,
    [switch]$Optional,
    [switch]$AdoptExisting,
    [switch]$RetryDegraded,
    [hashtable]$Details = @{}
  )

  $checkpoint = Read-StageCheckpoint -StateRoot $StateRoot -Name $Name
  $ready = Test-Readiness -Test $Test
  $fingerprintMatches = $checkpoint -and ([string]$checkpoint.fingerprint -eq $Fingerprint)

  if ($Optional -and $fingerprintMatches -and $checkpoint -and [string]$checkpoint.status -eq 'degraded' -and -not $RetryDegraded) {
    Write-Host "[resume] Optional stage $Name is recorded as degraded; skipping. Use RETRY-OPTIONAL.cmd to retry it." -ForegroundColor DarkYellow
    return $false
  }

  if ($ready -and $fingerprintMatches) {
    if (-not $checkpoint -or [string]$checkpoint.status -ne 'completed') {
      $priorAttempts = if ($checkpoint) { [int]$checkpoint.attempts } else { 0 }
      Write-StageCheckpoint -StateRoot $StateRoot -Name $Name -Fingerprint $Fingerprint -Status 'completed' -Attempts $priorAttempts -Details $Details
    }
    Write-Host "[resume] $Name already complete; skipping." -ForegroundColor DarkGreen
    return $true
  }

  # Adopt externally-completed work after an interrupted run. The readiness
  # probe must be strong enough to prove the stage; a stale marker alone never
  # counts as success.
  if ($ready -and -not $checkpoint -and $AdoptExisting) {
    Write-Host "[resume] $Name was already ready; recording checkpoint." -ForegroundColor DarkGreen
    Write-StageCheckpoint -StateRoot $StateRoot -Name $Name -Fingerprint $Fingerprint -Status 'completed' -Attempts 0 -Details $Details
    return $true
  }

  $attempts = if ($checkpoint -and $checkpoint.attempts) { [int]$checkpoint.attempts + 1 } else { 1 }
  $started = Get-Date -Format o
  Write-StageCheckpoint -StateRoot $StateRoot -Name $Name -Fingerprint $Fingerprint -Status 'running' -Attempts $attempts -StartedAt $started -Details $Details
  Write-Host "[run] $Name (attempt $attempts)" -ForegroundColor Yellow

  try {
    & $Action | Out-Host
    if (-not (Test-Readiness -Test $Test)) {
      throw "Stage '$Name' action finished but its readiness check did not pass."
    }
    Write-StageCheckpoint -StateRoot $StateRoot -Name $Name -Fingerprint $Fingerprint -Status 'completed' -Attempts $attempts -StartedAt $started -Details $Details
    Write-Host "[done] $Name" -ForegroundColor Green
    return $true
  } catch {
    $message = ($_ | Out-String).Trim()
    if ($Optional) {
      Write-StageCheckpoint -StateRoot $StateRoot -Name $Name -Fingerprint $Fingerprint -Status 'degraded' -ErrorText $message -Attempts $attempts -StartedAt $started -Details $Details
      Write-Warning "Optional stage '$Name' failed and was recorded as degraded. Core installation will continue. Use RETRY-OPTIONAL.cmd to retry it. $message"
      return $false
    }
    Write-StageCheckpoint -StateRoot $StateRoot -Name $Name -Fingerprint $Fingerprint -Status 'failed' -ErrorText $message -Attempts $attempts -StartedAt $started -Details $Details
    throw
  }
}

function Write-InstallSummary {
  param(
    [Parameter(Mandatory=$true)][string]$StateRoot,
    [Parameter(Mandatory=$true)][string]$OutputPath
  )
  $items = @()
  if (Test-Path $StateRoot) {
    foreach ($path in Get-ChildItem $StateRoot -Filter '*.json' -File | Sort-Object Name) {
      try { $items += Get-Content $path.FullName -Raw | ConvertFrom-Json } catch {}
    }
  }
  $summary = [ordered]@{
    generated_at = Get-Date -Format o
    completed = @($items | Where-Object status -eq 'completed').Count
    failed = @($items | Where-Object status -eq 'failed').Count
    degraded = @($items | Where-Object status -eq 'degraded').Count
    running = @($items | Where-Object status -eq 'running').Count
    stages = $items
  }
  Write-JsonUtf8NoBom -Path $OutputPath -Value $summary -Depth 12
}
