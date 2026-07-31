$ErrorActionPreference='Stop'
$AppRoot=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
Set-Location $AppRoot
if (-not (Get-Command opencode -ErrorAction SilentlyContinue)) { throw 'OpenCode CLI is not installed or not in PATH.' }
$health=Invoke-RestMethod http://127.0.0.1:8400/health
$studio=Invoke-RestMethod http://127.0.0.1:3001/api/settings
Write-Host "Worker lanes: $($health.lanes -join ', ')"
Write-Host 'Launching OpenCode with the 3D Gen Studio MCP control layer...'
opencode run --file (Join-Path $AppRoot 'prompts\opencode-build-full.md') "Inspect the current 3D Gen Studio project and continue the next single unproven pipeline stage."
