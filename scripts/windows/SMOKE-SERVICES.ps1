[CmdletBinding()]
param(
  [Parameter(Mandatory=$true)][string]$InstallRoot,
  [Parameter(Mandatory=$true)][string]$AppRoot,
  [Parameter(Mandatory=$true)][string]$StudioRoot,
  [Parameter(Mandatory=$true)][string]$ControlPython,
  [Parameter(Mandatory=$true)][string]$MeshPython,
  [Parameter(Mandatory=$true)][string]$ConfigPath,
  [Parameter(Mandatory=$true)][string]$OutputPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest
$logRoot = Join-Path $InstallRoot 'install-logs\smoke'
New-Item -ItemType Directory -Force -Path $logRoot,(Split-Path -Parent $OutputPath) | Out-Null

function Write-JsonUtf8NoBom([string]$Path,$Value,[int]$Depth=8) {
  $json = $Value | ConvertTo-Json -Depth $Depth
  [System.IO.File]::WriteAllText($Path, $json, (New-Object System.Text.UTF8Encoding($false)))
}

function Get-FreePort {
  $listener = New-Object System.Net.Sockets.TcpListener([System.Net.IPAddress]::Loopback, 0)
  $listener.Start()
  try { return ([System.Net.IPEndPoint]$listener.LocalEndpoint).Port }
  finally { $listener.Stop() }
}

function Test-Url([string]$Url) {
  try {
    Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3 | Out-Null
    return $true
  } catch { return $false }
}

function Wait-Url([string]$Name,[string]$Url,[System.Diagnostics.Process]$Process,[int]$Seconds) {
  $deadline = (Get-Date).AddSeconds($Seconds)
  do {
    if ($Process.HasExited) { throw "$Name exited before becoming ready (exit $($Process.ExitCode))." }
    if (Test-Url $Url) { return }
    Start-Sleep -Milliseconds 500
  } while ((Get-Date) -lt $deadline)
  throw "$Name did not become ready at $Url within $Seconds seconds."
}

function Start-SmokeProcess {
  param(
    [string]$Name,
    [string]$File,
    [string[]]$Arguments,
    [string]$WorkingDirectory,
    [hashtable]$Environment
  )
  $saved = @{}
  foreach ($key in $Environment.Keys) {
    $saved[$key] = [Environment]::GetEnvironmentVariable($key,'Process')
    [Environment]::SetEnvironmentVariable($key,[string]$Environment[$key],'Process')
  }
  try {
    $stdout = Join-Path $logRoot "$Name.stdout.log"
    $stderr = Join-Path $logRoot "$Name.stderr.log"
    return Start-Process -FilePath $File -ArgumentList $Arguments -WorkingDirectory $WorkingDirectory `
      -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
  } finally {
    foreach ($key in $Environment.Keys) {
      [Environment]::SetEnvironmentVariable($key,$saved[$key],'Process')
    }
  }
}

$processes = @()
$started = Get-Date
try {
  $meshPort = Get-FreePort
  $workerPort = Get-FreePort
  $studioPort = Get-FreePort

  $mesh = Start-SmokeProcess -Name 'meshtools' -File $MeshPython -Arguments @('main.py') `
    -WorkingDirectory (Join-Path $StudioRoot 'python-server') -Environment @{
      MESHTOOLS_HOST='127.0.0.1'; MESHTOOLS_PORT=$meshPort; MESHTOOLS_SKIP_GPU='1'
    }
  $processes += $mesh
  Wait-Url 'Mesh tools' "http://127.0.0.1:$meshPort/health" $mesh 90

  $worker = Start-SmokeProcess -Name 'worker' -File $ControlPython `
    -Arguments @('-m','uvicorn','service.app:app','--host','127.0.0.1','--port',[string]$workerPort) `
    -WorkingDirectory $AppRoot -Environment @{
      LOWVRAM3D_CONFIG=$ConfigPath; PYTHONPATH=(Join-Path $AppRoot 'src');
      HF_HOME=(Join-Path $InstallRoot 'models\huggingface')
    }
  $processes += $worker
  Wait-Url 'LowVRAM worker' "http://127.0.0.1:$workerPort/health" $worker 60

  $studio = Start-SmokeProcess -Name 'studio' -File 'node' -Arguments @('server.js') `
    -WorkingDirectory $StudioRoot -Environment @{ PORT=$studioPort }
  $processes += $studio
  Wait-Url '3D Gen Studio' "http://127.0.0.1:$studioPort/api/settings" $studio 90

  $result = [ordered]@{
    status = 'passed'
    started_at = $started.ToString('o')
    completed_at = (Get-Date -Format o)
    services = @(
      @{name='mesh-tools'; url="http://127.0.0.1:$meshPort/health"; pid=$mesh.Id},
      @{name='lowvram-worker'; url="http://127.0.0.1:$workerPort/health"; pid=$worker.Id},
      @{name='3d-gen-studio'; url="http://127.0.0.1:$studioPort/api/settings"; pid=$studio.Id}
    )
    logs = $logRoot
  }
  Write-JsonUtf8NoBom -Path $OutputPath -Value $result -Depth 8
} catch {
  $result = [ordered]@{
    status = 'failed'
    started_at = $started.ToString('o')
    failed_at = (Get-Date -Format o)
    error = ($_ | Out-String).Trim()
    logs = $logRoot
  }
  Write-JsonUtf8NoBom -Path $OutputPath -Value $result -Depth 8
  throw
} finally {
  foreach ($process in $processes) {
    if ($process -and -not $process.HasExited) {
      & taskkill.exe /PID $process.Id /T /F | Out-Null
    }
  }
}
