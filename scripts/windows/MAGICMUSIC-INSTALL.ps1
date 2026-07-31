$ErrorActionPreference='Stop'
$PackageRoot=Split-Path -Parent (Split-Path -Parent $MyInvocation.MyCommand.Path)
$Installer=Join-Path $PackageRoot 'INSTALL-ONE-CLICK.ps1'
if (-not (Test-Path $Installer)) { throw "Installer missing: $Installer" }
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Installer
if ($LASTEXITCODE -ne 0) { throw "Installer failed with exit code $LASTEXITCODE" }
