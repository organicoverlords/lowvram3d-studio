[CmdletBinding()]
param(
    [string]$JobId = "",
    [ValidateRange(512, 2048)]
    [int]$AtlasSize = 512,
    # views/projection carries pixel-verified renders and ships its own view_metadata.json.
    # views/mv_adapter is NOT a safe default: on job 26a37e41 all six of its PNGs are byte-identical
    # pure black, because the SD2.1 latents went non-finite on Turing fp16 and diffusers cast NaN to
    # 0 while the worker still recorded "success": true. Texturing from those yields a structurally
    # valid, entirely black GLB, so the view set is named explicitly and verified before use.
    [string]$ViewsSubdir = "views/projection"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallRoot = Join-Path $env:LOCALAPPDATA "LowVRAM3DStudio"
$JobsRoot = Join-Path $InstallRoot "jobs"
$PythonCandidates = @(
    $env:LOWVRAM3D_CONTROL_PYTHON,
    (Join-Path $InstallRoot "envs\control\Scripts\python.exe")
)
$BlenderCandidates = @(
    $env:LOWVRAM3D_BLENDER,
    "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    "C:\Program Files\Blender Foundation\Blender 4.2\blender.exe"
)
$ArtifactDir = if ($env:RUNNER_TEMP) {
    Join-Path $env:RUNNER_TEMP "shaman-textured-baseline"
} else {
    Join-Path $env:TEMP "shaman-textured-baseline"
}

function Resolve-ExistingFile([object[]]$Candidates, [string]$Label) {
    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath ([string]$candidate) -PathType Leaf)) {
            return (Resolve-Path -LiteralPath ([string]$candidate)).Path
        }
    }
    throw "Could not locate $Label. Checked: $($Candidates -join '; ')"
}

function Test-BaselineJob([string]$CandidateDir) {
    $views = Join-Path $CandidateDir ($ViewsSubdir -replace '/', '\')
    if (-not (Test-Path -LiteralPath (Join-Path $views "view_metadata.json") -PathType Leaf)) {
        return $false
    }
    foreach ($view in @("front", "right", "back", "left")) {
        if (-not (Test-Path -LiteralPath (Join-Path $views "$view.png") -PathType Leaf)) { return $false }
    }
    return (
        (Test-Path -LiteralPath (Join-Path $CandidateDir "uv\game_ready_uv.glb") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir "proof\job_receipt.json") -PathType Leaf)
    )
}

if (-not (Test-Path -LiteralPath $JobsRoot -PathType Container)) {
    throw "LowVRAM3D jobs directory is missing: $JobsRoot"
}

if (-not $JobId) {
    $candidate = Get-ChildItem -LiteralPath $JobsRoot -Directory -ErrorAction Stop |
        Where-Object { Test-BaselineJob $_.FullName } |
        Sort-Object LastWriteTimeUtc -Descending |
        Select-Object -First 1
    if (-not $candidate) {
        throw "No job contains a receipt, game_ready_uv.glb and a complete $ViewsSubdir view set."
    }
    $JobId = $candidate.Name
}

$JobDir = Join-Path $JobsRoot $JobId
if (-not (Test-BaselineJob $JobDir)) {
    throw "Job $JobId is not ready for textured-baseline recovery: $JobDir"
}

$python = Resolve-ExistingFile $PythonCandidates "LowVRAM3D control Python"
$blender = Resolve-ExistingFile $BlenderCandidates "Blender executable"

Write-Host "Recovering textured baseline from job $JobId" -ForegroundColor Cyan
Write-Host "UV mesh: $(Join-Path $JobDir 'uv\game_ready_uv.glb')"
Write-Host "Views:   $(Join-Path $JobDir ($ViewsSubdir -replace '/', '\'))"

# A cancelled Actions job can leave Blender's slow project_texture.py child alive. Terminate only
# processes whose command line names this exact job, so unrelated Blender or Python work is kept.
$targeted = Get-CimInstance Win32_Process -ErrorAction SilentlyContinue | Where-Object {
    $_.ProcessId -ne $PID -and
    $_.CommandLine -and
    $_.CommandLine.Contains($JobId) -and
    ($_.Name -ieq "blender.exe" -or $_.Name -ieq "python.exe")
}
foreach ($process in $targeted) {
    Write-Host "Stopping stale job process PID=$($process.ProcessId) NAME=$($process.Name)" -ForegroundColor Yellow
    Stop-Process -Id $process.ProcessId -Force -ErrorAction SilentlyContinue
}
if ($targeted) { Start-Sleep -Seconds 2 }

if (Test-Path -LiteralPath $ArtifactDir) {
    Remove-Item -LiteralPath $ArtifactDir -Recurse -Force
}
New-Item -ItemType Directory -Path $ArtifactDir -Force | Out-Null

& $python (Join-Path $RepoRoot "workers\recover_textured_baseline.py") `
    --job-dir $JobDir `
    --views-subdir $ViewsSubdir `
    --output-dir $ArtifactDir `
    --atlas-size $AtlasSize `
    --blender $blender
if ($LASTEXITCODE -ne 0) {
    throw "Textured-baseline recovery failed with exit code $LASTEXITCODE"
}

$glb = Join-Path $ArtifactDir "shaman_textured_baseline.glb"
if (-not (Test-Path -LiteralPath $glb -PathType Leaf)) {
    throw "Recovery returned success without the textured GLB: $glb"
}
if ((Get-Item -LiteralPath $glb).Length -le 0) {
    throw "Recovered textured GLB is empty: $glb"
}

# Independent proof in a fresh Blender process. The producing run's own word is not evidence:
# it must reimport, carry UVs, resolve a packed base-colour texture, and render a preview.
$validation = Join-Path $ArtifactDir "shaman_textured_baseline_validation.json"
$preview = Join-Path $ArtifactDir "shaman_textured_baseline_front_preview.png"
$env:PYTHONPATH = "$RepoRoot\blender;$RepoRoot\src"
& $blender --background --python-use-system-env `
    --python (Join-Path $RepoRoot "blender\validate_textured_baseline.py") -- `
    --glb $glb --report $validation --preview $preview
if ($LASTEXITCODE -ne 0) {
    throw "Fresh-Blender validation failed with exit code $LASTEXITCODE"
}
if (-not (Test-Path -LiteralPath $validation -PathType Leaf)) {
    throw "Validation produced no report: $validation"
}
$verdict = Get-Content -LiteralPath $validation -Raw | ConvertFrom-Json
if (-not $verdict.success) {
    throw "Validation report records failure: $($verdict | ConvertTo-Json -Depth 6)"
}
if (-not (Test-Path -LiteralPath $preview -PathType Leaf)) {
    throw "Validation produced no front preview: $preview"
}

$hash = (Get-FileHash -LiteralPath $glb -Algorithm SHA256).Hash
Set-Content -LiteralPath (Join-Path $ArtifactDir "source_job_id.txt") -Value $JobId -Encoding utf8
Write-Host "glb=$glb bytes=$((Get-Item -LiteralPath $glb).Length) sha256=$hash"
Write-Host "TEXTURED_BASELINE_PROVEN job=$JobId" -ForegroundColor Green
