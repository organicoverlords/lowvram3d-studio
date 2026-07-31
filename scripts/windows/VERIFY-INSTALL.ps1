$ErrorActionPreference='Stop'
$AppRoot=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$InstallRoot=Split-Path -Parent $AppRoot
$config=Get-Content (Join-Path $AppRoot 'config\local.json') -Raw | ConvertFrom-Json
$python=Join-Path $InstallRoot 'envs\control\Scripts\python.exe'
$env:PYTHONPATH=Join-Path $AppRoot 'src'
& $python -m compileall -q (Join-Path $AppRoot 'src') (Join-Path $AppRoot 'service') (Join-Path $AppRoot 'workers') (Join-Path $AppRoot 'scripts')
& $python -m unittest discover -s (Join-Path $AppRoot 'tests') -v
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8400/health | Select-Object -Expand Content
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:8200/health | Select-Object -Expand Content
Invoke-WebRequest -UseBasicParsing http://127.0.0.1:3001/api/settings | Out-Null
Write-Host 'Installer and services verified.' -ForegroundColor Green
