[CmdletBinding()]
param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$AppRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$InstallRoot = Split-Path -Parent $AppRoot
$ConfigPath = Join-Path $AppRoot 'config\local.json'
$PidDir = Join-Path $InstallRoot 'pids'
$LogDir = Join-Path $InstallRoot 'runtime-logs'
$ProofDir = Join-Path $InstallRoot 'proof'
New-Item -ItemType Directory -Force -Path $PidDir,$LogDir,$ProofDir | Out-Null
if (-not (Test-Path $ConfigPath)) { throw "Run INSTALL-ONE-CLICK.cmd first. Missing $ConfigPath" }
$config = Get-Content $ConfigPath -Raw | ConvertFrom-Json
$ControlPython = Join-Path $InstallRoot 'envs\control\Scripts\python.exe'
$StudioRoot = [string]$config.extra.studio_root
$MeshPython = Join-Path $StudioRoot 'python-server\.venv\Scripts\python.exe'
$started = @()

function Write-JsonUtf8NoBom([string]$Path,$Value,[int]$Depth=8) {
  $json = $Value | ConvertTo-Json -Depth $Depth
  [System.IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding($false)))
}

function Test-Url([string]$Url) {
  try { Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3 | Out-Null; return $true }
  catch { return $false }
}

function Wait-Url([string]$Url, [int]$Seconds=120, [System.Diagnostics.Process]$Process=$null) {
  $until = (Get-Date).AddSeconds($Seconds)
  do {
    if ($Process -and $Process.HasExited) { throw "Service exited before becoming ready: $Url (exit $($Process.ExitCode))" }
    if (Test-Url $Url) { return }
    Start-Sleep -Milliseconds 750
  } while ((Get-Date) -lt $until)
  throw "Service did not start: $Url"
}

function Read-PidRecord([string]$Path) {
  if (-not (Test-Path $Path)) { return $null }
  $raw = (Get-Content $Path -Raw -ErrorAction SilentlyContinue).Trim()
  if (-not $raw) { return $null }
  try { return $raw | ConvertFrom-Json }
  catch {
    $legacy = 0
    if ([int]::TryParse($raw, [ref]$legacy)) { return [pscustomobject]@{ pid=$legacy; legacy=$true } }
    return $null
  }
}

function Test-RecordMatchesProcess($Record,[System.Diagnostics.Process]$Process) {
  if (-not $Record -or -not $Process) { return $false }
  if (-not ($Record.PSObject.Properties.Name -contains 'process_start_time')) { return $true }
  try {
    $recorded = [DateTime]::Parse([string]$Record.process_start_time).ToUniversalTime()
    $actual = $Process.StartTime.ToUniversalTime()
    return ([Math]::Abs(($actual - $recorded).TotalSeconds) -lt 5)
  } catch { return $false }
}

function Stop-TrackedRecord([string]$PidFile) {
  $record = Read-PidRecord $PidFile
  if ($record -and [int]$record.pid -gt 0) {
    $process = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if ($process -and (Test-RecordMatchesProcess $record $process)) {
      & taskkill.exe /PID ([int]$record.pid) /T /F | Out-Null
    }
  }
  Remove-Item $PidFile -Force -ErrorAction SilentlyContinue
}

function Start-Tracked {
  [CmdletBinding()]
  param(
    [Parameter(Mandatory=$true)][string]$Name,
    [Parameter(Mandatory=$true)][string]$File,
    [Parameter(Mandatory=$true)][string[]]$Arguments,
    [Parameter(Mandatory=$true)][string]$WorkingDirectory,
    [Parameter(Mandatory=$true)][string]$HealthUrl,
    [hashtable]$Environment=@{},
    [switch]$NoRedirect
  )
  if (Test-Url $HealthUrl) {
    Write-Host "[ready] $Name already answers at $HealthUrl" -ForegroundColor DarkGreen
    return $null
  }

  $pidFile = Join-Path $PidDir "$Name.pid"
  $record = Read-PidRecord $pidFile
  if ($record -and [int]$record.pid -gt 0) {
    $old = Get-Process -Id ([int]$record.pid) -ErrorAction SilentlyContinue
    if ($old) {
      Write-Host "[resume] Waiting briefly for tracked $Name process $($record.pid)..." -ForegroundColor DarkGray
      try { Wait-Url $HealthUrl 10 $old; return $null }
      catch {
        Write-Warning "Tracked $Name process is unhealthy; terminating only its recorded process tree."
        Stop-TrackedRecord $pidFile
      }
    } else { Remove-Item $pidFile -Force -ErrorAction SilentlyContinue }
  }

  $saved = @{}
  foreach ($key in $Environment.Keys) {
    $saved[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
    [Environment]::SetEnvironmentVariable($key,[string]$Environment[$key],'Process')
  }
  try {
    if ($NoRedirect) {
      $process = Start-Process -FilePath $File -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory -WindowStyle Minimized -PassThru
    } else {
      $stdout = Join-Path $LogDir "$Name.stdout.log"
      $stderr = Join-Path $LogDir "$Name.stderr.log"
      $process = Start-Process -FilePath $File -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory `
        -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
    }
  } finally {
    foreach ($key in $Environment.Keys) {
      [Environment]::SetEnvironmentVariable($key,$saved[$key],'Process')
    }
  }
  $pidRecord = [ordered]@{
    pid = $process.Id
    name = $Name
    started_at = Get-Date -Format o
    process_start_time = $process.StartTime.ToUniversalTime().ToString('o')
    file = $File
    arguments = $Arguments
    health_url = $HealthUrl
  }
  Write-JsonUtf8NoBom -Path $pidFile -Value $pidRecord -Depth 6
  $script:started += [pscustomobject]@{ Name=$Name; Process=$process; PidFile=$pidFile }
  return $process
}

try {
  # Existing ComfyUI is reused. Its console streams are intentionally not redirected:
  # that launch pattern previously triggered a Windows forrtl window-CLOSE abort.
  $comfyHealth = "$($config.comfyui_url)/system_stats"
  if (-not (Test-Url $comfyHealth) -and $config.comfyui_path) {
    $comfyPython = [string]$config.extra.comfyui_python
    if (-not $comfyPython) { $comfyPython = 'python' }
    $comfyArgs = @('-s',(Join-Path $config.comfyui_path 'main.py'),'--windows-standalone-build','--lowvram')
    if ($comfyPython -eq 'python') { $comfyArgs = @((Join-Path $config.comfyui_path 'main.py'),'--lowvram') }
    $comfyProcess = Start-Tracked -Name 'comfyui' -File $comfyPython -Arguments $comfyArgs `
      -WorkingDirectory $config.comfyui_path -HealthUrl $comfyHealth `
      -Environment @{'PYTORCH_CUDA_ALLOC_CONF'='expandable_segments:True'} -NoRedirect
    Wait-Url $comfyHealth 180 $comfyProcess
  }

  $meshHealth = 'http://127.0.0.1:8200/health'
  $meshProcess = Start-Tracked -Name 'meshtools' -File $MeshPython -Arguments @('main.py') `
    -WorkingDirectory (Join-Path $StudioRoot 'python-server') -HealthUrl $meshHealth `
    -Environment @{'MESHTOOLS_SKIP_GPU'='1'; 'MESHTOOLS_HOST'='127.0.0.1'; 'MESHTOOLS_PORT'='8200'}
  Wait-Url $meshHealth 120 $meshProcess

  $workerHealth = 'http://127.0.0.1:8400/health'
  $workerProcess = Start-Tracked -Name 'lowvram-worker' -File $ControlPython `
    -Arguments @('-m','uvicorn','service.app:app','--host','127.0.0.1','--port','8400') `
    -WorkingDirectory $AppRoot -HealthUrl $workerHealth -Environment @{
      'LOWVRAM3D_CONFIG'=$ConfigPath
      'PYTHONPATH'=(Join-Path $AppRoot 'src')
      'HF_HOME'=(Join-Path $InstallRoot 'models\huggingface')
    }
  Wait-Url $workerHealth 60 $workerProcess

  $studioHealth = 'http://127.0.0.1:8311/api/settings'
  $studioProcess = Start-Tracked -Name '3d-gen-studio' -File 'node' -Arguments @('server.js') `
    -WorkingDirectory $StudioRoot -HealthUrl $studioHealth -Environment @{'PORT'='8311'}
  Wait-Url $studioHealth 120 $studioProcess

  $registered = Join-Path $ProofDir 'studio-registration.json'
  $regArgs = @(
    (Join-Path $AppRoot 'scripts\register_studio.py'),
    '--studio-url','http://127.0.0.1:8311',
    '--worker-url','http://127.0.0.1:8400',
    '--comfy-path',[string]$config.comfyui_path,
    '--comfy-url',[string]$config.comfyui_url,
    '--output',$registered
  )
  if (-not (Test-Path $registered)) { $regArgs += '--create-project' }
  & $ControlPython @regArgs
  if ($LASTEXITCODE -ne 0) { throw '3D Gen Studio provider registration failed.' }

  Remove-Item (Join-Path $ProofDir 'RUNTIME_START_FAILED.txt') -Force -ErrorAction SilentlyContinue
  Write-Host 'LowVRAM 3D Studio is ready.' -ForegroundColor Green
  Write-Host 'Primary: Mini Turbo + MV-Adapter. Fallback 1: deterministic projection. Fallback 2: TripoSR when ready.'
  if (-not $NoBrowser) { Start-Process 'http://127.0.0.1:8311' }
} catch {
  ($_ | Out-String) | Set-Content -Encoding UTF8 (Join-Path $ProofDir 'RUNTIME_START_FAILED.txt')
  foreach ($item in $started) {
    if ($item.Process -and -not $item.Process.HasExited) { & taskkill.exe /PID $item.Process.Id /T /F | Out-Null }
    Remove-Item $item.PidFile -Force -ErrorAction SilentlyContinue
  }
  Write-Host 'Startup failed; only processes started by this attempt were stopped.' -ForegroundColor Yellow
  throw
}
