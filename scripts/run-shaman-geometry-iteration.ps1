[CmdletBinding()]
param(
    [string]$ImagePath = "C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png",
    [string]$ExpectedSha256 = "4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f",
    [ValidateRange(50000, 1000000)]
    [int]$PreserveFaces = 350000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallRoot = Join-Path $env:LOCALAPPDATA "LowVRAM3DStudio"
$CanonicalRoot = "C:\AI\LowVRAM3D-benchmarks"
$CanonicalImage = Join-Path $CanonicalRoot "images\antlered_bird_shaman_anchor.png"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OutputDir = Join-Path $CanonicalRoot "outputs\antlered_bird_shaman_anchor\geometry-iterations\$RunStamp"
$LatestDir = Join-Path $CanonicalRoot "outputs\antlered_bird_shaman_anchor\geometry-latest"
$ArtifactDir = if ($env:RUNNER_TEMP) {
    Join-Path $env:RUNNER_TEMP "shaman-geometry-iteration"
} else {
    Join-Path $OutputDir "workflow-artifact"
}

function Resolve-ExistingFile([object[]]$Candidates, [string]$Label) {
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath ([string]$candidate) -PathType Leaf)) {
            return (Resolve-Path -LiteralPath ([string]$candidate)).Path
        }
    }
    throw "Could not locate $Label. Checked: $($Candidates -join '; ')"
}

function Resolve-ControlPython {
    return Resolve-ExistingFile @(
        $env:LOWVRAM3D_CONTROL_PYTHON,
        (Join-Path $InstallRoot "envs\control\Scripts\python.exe")
    ) "LowVRAM3D control Python"
}

function Resolve-PipelineConfig {
    $candidates = @(
        $env:LOWVRAM3D_CONFIG,
        (Join-Path $RepoRoot "config\local.json"),
        (Join-Path $InstallRoot "app\config\local.json"),
        (Join-Path $InstallRoot "config\local.json"),
        (Join-Path $InstallRoot "local.json")
    )
    foreach ($candidate in $candidates) {
        if ($candidate -and (Test-Path -LiteralPath ([string]$candidate) -PathType Leaf)) {
            return (Resolve-Path -LiteralPath ([string]$candidate)).Path
        }
    }
    $found = Get-ChildItem -LiteralPath $InstallRoot -Filter local.json -File -Recurse -ErrorAction SilentlyContinue |
        Select-Object -First 1
    if ($found) { return $found.FullName }
    throw "Could not locate LowVRAM3D config/local.json."
}

function Copy-DiagnosticDirectory([string]$Source, [string]$DestinationName) {
    if (-not $Source -or -not (Test-Path -LiteralPath $Source -PathType Container)) { return }
    Copy-Item -LiteralPath $Source -Destination (Join-Path $ArtifactDir $DestinationName) -Recurse -Force -ErrorAction SilentlyContinue
}

if (-not (Test-Path -LiteralPath $ImagePath -PathType Leaf)) {
    throw "Shaman source image is missing: $ImagePath"
}
$actualHash = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "Shaman image SHA-256 mismatch. Expected $ExpectedSha256, got $actualHash"
}

# Remove only the obsolete detached MagicMusic pipeline from the rejected fast-baseline route.
$obsolete = @(
    Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object {
            $_.CommandLine -and
            $_.CommandLine.Contains("run_full_image.py") -and
            $_.CommandLine.Contains("antlered_bird_shaman_anchor") -and
            $_.CommandLine.Contains("fast-baseline")
        }
)
foreach ($process in $obsolete) {
    Write-Host "Stopping obsolete MagicMusic pipeline PID=$($process.ProcessId)" -ForegroundColor Yellow
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($obsolete.Count -gt 0) { Start-Sleep -Seconds 3 }

New-Item -ItemType Directory -Path (Split-Path $CanonicalImage) -Force | Out-Null
if (-not (Test-Path -LiteralPath $CanonicalImage) -or
    (Get-FileHash -LiteralPath $CanonicalImage -Algorithm SHA256).Hash.ToLowerInvariant() -ne $actualHash) {
    Copy-Item -LiteralPath $ImagePath -Destination $CanonicalImage -Force
}

foreach ($directory in @($OutputDir, $ArtifactDir)) {
    if (Test-Path -LiteralPath $directory) {
        Remove-Item -LiteralPath $directory -Recurse -Force
    }
    New-Item -ItemType Directory -Path $directory -Force | Out-Null
}

$python = Resolve-ControlPython
$config = Resolve-PipelineConfig
$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
$env:PYTHONUNBUFFERED = "1"

# Mini Turbo is currently failing to return a mesh on this installation. The bounded geometry
# iteration therefore selects the installed TripoSR lane explicitly and raises only its marching-
# cubes detail. Normal production fallback behaviour remains unchanged outside this workflow.
$env:LOWVRAM3D_PROXY_BACKEND = "triposr"
$env:LOWVRAM3D_TRIPOSR_MC = "384"
$env:LOWVRAM3D_TRIPOSR_CHUNK = "1024"
Write-Host "Geometry fallback: TripoSR mc=$env:LOWVRAM3D_TRIPOSR_MC chunk=$env:LOWVRAM3D_TRIPOSR_CHUNK" -ForegroundColor Cyan

$prompt = @"
Antlered bird-shaman character. Preserve the source silhouette, upright body, avian head and beak,
broad branching antlers, layered feathers and robes, ornaments, separate arms and legs, and the
complete staff. Produce coherent high-detail geometry with readable anatomy and separated major
forms. No horizontal collapse, merged staff, duplicate limbs, floating debris or missing antlers.
"@ -replace "\s+", " "

& $python (Join-Path $RepoRoot "workers\run_geometry_iteration.py") `
    --config $config `
    --image $CanonicalImage `
    --expected-image-sha256 $ExpectedSha256 `
    --output-dir $OutputDir `
    --prompt $prompt `
    --preserve-faces $PreserveFaces
$exitCode = $LASTEXITCODE

$resultPath = Join-Path $OutputDir "geometry_iteration_result.json"
Copy-Item -LiteralPath $resultPath -Destination $ArtifactDir -Force -ErrorAction SilentlyContinue
if (Test-Path -LiteralPath (Join-Path $OutputDir "deliverable")) {
    Copy-Item -LiteralPath (Join-Path $OutputDir "deliverable") -Destination (Join-Path $ArtifactDir "deliverable") -Recurse -Force
}

if (Test-Path -LiteralPath $resultPath) {
    try {
        $result = Get-Content -LiteralPath $resultPath -Raw | ConvertFrom-Json
        $jobDir = [string]$result.job_dir
        if ($jobDir) {
            Set-Content -LiteralPath (Join-Path $ArtifactDir "job_dir.txt") -Value $jobDir -Encoding utf8
            Copy-DiagnosticDirectory (Join-Path $jobDir "logs") "job-logs"
            Copy-DiagnosticDirectory (Join-Path $jobDir "proof") "job-proof"
            Copy-DiagnosticDirectory (Join-Path $jobDir "preprocess") "job-preprocess"
        }
    } catch {
        Set-Content -LiteralPath (Join-Path $ArtifactDir "diagnostic-copy-error.txt") -Value $_.Exception.ToString() -Encoding utf8
    }
}

if ($exitCode -ne 0) {
    throw "Shaman geometry iteration failed with exit code $exitCode. See $resultPath"
}

$master = Join-Path $OutputDir "deliverable\shaman_geometry_master.glb"
$working = Join-Path $OutputDir "deliverable\shaman_geometry_working.glb"
$validation = Join-Path $OutputDir "deliverable\geometry_validation.json"
foreach ($required in @($master, $working, $validation)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf) -or (Get-Item -LiteralPath $required).Length -le 0) {
        throw "Geometry iteration returned success without required artifact: $required"
    }
}

if (Test-Path -LiteralPath $LatestDir) {
    Remove-Item -LiteralPath $LatestDir -Recurse -Force
}
New-Item -ItemType Directory -Path $LatestDir -Force | Out-Null
Copy-Item -LiteralPath (Join-Path $OutputDir "deliverable\*") -Destination $LatestDir -Recurse -Force
Copy-Item -LiteralPath $resultPath -Destination $LatestDir -Force

$masterHash = (Get-FileHash -LiteralPath $master -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "SHAMAN_GEOMETRY_ITERATION_READY" -ForegroundColor Green
Write-Host "MASTER: $master"
Write-Host "LATEST: $LatestDir"
Write-Host "SHA256: $masterHash"
Write-Host "RESULT: $resultPath"
