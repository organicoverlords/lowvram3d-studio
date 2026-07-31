[CmdletBinding()]
param(
    [string]$ImagePath = "C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png",
    [string]$ExpectedSha256 = "4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CanonicalRoot = "C:\AI\LowVRAM3D-benchmarks"
$CanonicalImage = Join-Path $CanonicalRoot "images\antlered_bird_shaman_anchor.png"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OutputDir = Join-Path $CanonicalRoot "outputs\antlered_bird_shaman_anchor\runs\$RunStamp"
$ArtifactDir = if ($env:RUNNER_TEMP) { Join-Path $env:RUNNER_TEMP "shaman-image-to-3d" } else { Join-Path $OutputDir "workflow-artifact" }

function Resolve-ControlPython {
    $candidates = @(
        $env:LOWVRAM3D_CONTROL_PYTHON,
        "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return (Resolve-Path $candidate).Path }
    }
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($python) { return $python.Source }
    throw "No LowVRAM3D control Python or python.exe was found."
}

function Resolve-PipelineConfig {
    $candidates = @(
        $env:LOWVRAM3D_CONFIG,
        (Join-Path $RepoRoot "config\local.json"),
        "$env:LOCALAPPDATA\LowVRAM3DStudio\config\local.json",
        "$env:LOCALAPPDATA\LowVRAM3DStudio\local.json",
        "C:\AI\LowVRAM3DStudio\config\local.json"
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path $candidate)) { return (Resolve-Path $candidate).Path }
    }
    $installRoot = "$env:LOCALAPPDATA\LowVRAM3DStudio"
    if (Test-Path $installRoot) {
        $found = Get-ChildItem $installRoot -Filter local.json -File -Recurse -ErrorAction SilentlyContinue | Select-Object -First 1
        if ($found) { return $found.FullName }
    }
    throw "Could not locate LowVRAM3D config/local.json."
}

function Get-ImageDimensions([string]$Path) {
    Add-Type -AssemblyName System.Drawing
    $image = [System.Drawing.Image]::FromFile($Path)
    try {
        return [ordered]@{ width = $image.Width; height = $image.Height; format = $image.RawFormat.Guid.ToString() }
    } finally {
        $image.Dispose()
    }
}

if (-not (Test-Path -LiteralPath $ImagePath)) { throw "Shaman source image is missing: $ImagePath" }
$actualHash = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "Shaman image SHA-256 mismatch. Expected $ExpectedSha256, got $actualHash"
}
$dimensions = Get-ImageDimensions $ImagePath
if ($dimensions.width -ne 1122 -or $dimensions.height -ne 1402) {
    throw "Shaman image dimensions mismatch. Expected 1122x1402, got $($dimensions.width)x$($dimensions.height)"
}

New-Item -ItemType Directory -Path (Split-Path $CanonicalImage) -Force | Out-Null
if (-not (Test-Path $CanonicalImage) -or (Get-FileHash $CanonicalImage -Algorithm SHA256).Hash.ToLowerInvariant() -ne $actualHash) {
    Copy-Item -LiteralPath $ImagePath -Destination $CanonicalImage -Force
}
New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
New-Item -ItemType Directory -Path $ArtifactDir -Force | Out-Null

$python = Resolve-ControlPython
$config = Resolve-PipelineConfig
$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"

$hardware = [ordered]@{
    started_at = (Get-Date).ToString("o")
    computer = $env:COMPUTERNAME
    runner = $env:RUNNER_NAME
    repository_head = (& git -C $RepoRoot rev-parse HEAD).Trim()
    image = $ImagePath
    image_sha256 = $actualHash
    image_width = $dimensions.width
    image_height = $dimensions.height
    canonical_image = $CanonicalImage
    previous_rejected_sha256 = "eccef854f816f446ce2bf2e08559df519adff223b425c1dccc1c0a9b299f13f6"
    python = $python
    config = $config
    gpu = @()
}
$nvidia = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($nvidia) { $hardware.gpu = @(& $nvidia.Source --query-gpu=name,driver_version,memory.total,memory.used --format=csv,noheader,nounits) }
$hardware | ConvertTo-Json -Depth 6 | Set-Content (Join-Path $ArtifactDir "hardware.json") -Encoding utf8

$prompt = "Antlered bird-shaman character, preserve the exact silhouette, beak, antlers, layered feathers, ornaments, staff and clothing from the source image; photorealistic detailed game-ready character, clear material separation, no duplicate limbs or debris."

& $python (Join-Path $RepoRoot "workers\run_full_image.py") `
    --config $config `
    --image $CanonicalImage `
    --expected-image-sha256 $ExpectedSha256 `
    --output-dir $OutputDir `
    --prompt $prompt `
    --asset-type character `
    --quality hero `
    --texture-resolution 2048 `
    --animation-preset idle
$exitCode = $LASTEXITCODE

Copy-Item (Join-Path $OutputDir "run_result.json") $ArtifactDir -Force -ErrorAction SilentlyContinue
if (Test-Path (Join-Path $OutputDir "deliverable")) {
    Copy-Item (Join-Path $OutputDir "deliverable") (Join-Path $ArtifactDir "deliverable") -Recurse -Force
}

if ($exitCode -ne 0) { throw "Shaman image-to-3D pipeline failed with exit code $exitCode. See $OutputDir\run_result.json" }
Write-Host "SHAMAN_RUN_PASSED output=$OutputDir artifact=$ArtifactDir" -ForegroundColor Green
