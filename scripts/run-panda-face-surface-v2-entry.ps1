param(
    [Parameter(Mandatory = $true)][string]$RunId,
    [Parameter(Mandatory = $true)][string]$Repository,
    [Parameter(Mandatory = $true)][string]$Branch,
    [Parameter(Mandatory = $true)][string]$HeadSha
)

$ErrorActionPreference = "Stop"
$pythonDirectory = "C:\AI\HY3D2\python_standalone"
$blenderDirectory = "C:\Program Files\Blender Foundation\Blender 5.2"
foreach ($directory in @($pythonDirectory, $blenderDirectory)) {
    if (-not (Test-Path -LiteralPath $directory -PathType Container)) {
        throw "Required tool directory is missing: $directory"
    }
}
$env:PATH = "$pythonDirectory;$blenderDirectory;$env:PATH"
$env:PYTHONWARNINGS = "ignore::DeprecationWarning"
$implementation = Join-Path $PSScriptRoot "run-panda-face-surface-v2.ps1"
& powershell.exe -NoProfile -ExecutionPolicy Bypass -File $implementation `
    -RunId $RunId `
    -Repository $Repository `
    -Branch $Branch `
    -HeadSha $HeadSha
exit $LASTEXITCODE
