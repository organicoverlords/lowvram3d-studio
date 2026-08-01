[CmdletBinding()]
param(
    [string]$Image = "",
    [string]$Manifest = "",
    [string]$Profile = "Auto",
    [string]$AssetId = "",
    [string]$OutputRoot = "",
    [string]$FromStage = "INGEST",
    [string]$ToStage = "TEXTURE_QA",
    [string]$ExistingMaster = "",
    [string]$PythonPath = "C:\AI\HY3D2\python_standalone\python.exe",
    [string]$BlenderPath = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    [switch]$WriteManifestOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Runner = Join-Path $RepoRoot "workers\run_asset_pipeline.py"

if (-not (Test-Path -LiteralPath $Runner)) { throw "Pipeline runner is missing: $Runner" }
if (-not $Image -and -not $Manifest) { throw "Provide -Image or -Manifest." }
if ($Image -and -not (Test-Path -LiteralPath $Image)) { throw "Source image not found: $Image" }

if (-not (Test-Path -LiteralPath $PythonPath)) {
    $candidate = Get-Command python.exe -ErrorAction SilentlyContinue
    if (-not $candidate) { throw "Python was not found at '$PythonPath' or on PATH." }
    $PythonPath = $candidate.Source
}
if (-not (Test-Path -LiteralPath $BlenderPath)) {
    $candidate = Get-Command blender.exe -ErrorAction SilentlyContinue
    if ($candidate) { $BlenderPath = $candidate.Source }
}

$env:PYTHONPATH = "$RepoRoot\blender;$RepoRoot\src;$RepoRoot\workers;$RepoRoot"

$arguments = @($Runner, "--python", $PythonPath, "--blender", $BlenderPath,
               "--from-stage", $FromStage, "--to-stage", $ToStage)
if ($Image)          { $arguments += @("--image", $Image) }
if ($Manifest)       { $arguments += @("--manifest", $Manifest) }
if ($Profile)        { $arguments += @("--profile", $Profile) }
if ($AssetId)        { $arguments += @("--asset-id", $AssetId) }
if ($OutputRoot)     { $arguments += @("--output-root", $OutputRoot) }
if ($ExistingMaster) { $arguments += @("--existing-master", $ExistingMaster) }
if ($WriteManifestOnly) { $arguments += "--write-manifest-only" }

Write-Host "ASSET_PIPELINE_START" -ForegroundColor Cyan
& $PythonPath @arguments
$exit = $LASTEXITCODE

if ($exit -ne 0) {
    Write-Host "ASSET_PIPELINE_FAILED (exit $exit)" -ForegroundColor Yellow
    # exit, not throw: a throw leaves $LASTEXITCODE wherever the child left it, so a caller
    # checking the exit code sails straight past a failed run.
    exit $exit
}
Write-Host "ASSET_PIPELINE_COMPLETED" -ForegroundColor Green
exit 0
