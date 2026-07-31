[CmdletBinding()]
param(
    [string]$JobId = "",
    [ValidateRange(512, 1024)]
    [int]$AtlasSize = 512
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$InstallRoot = Join-Path $env:LOCALAPPDATA "LowVRAM3DStudio"
$JobsRoot = Join-Path $InstallRoot "jobs"
$ConfigCandidates = @(
    $env:LOWVRAM3D_CONFIG,
    (Join-Path $RepoRoot "config\local.json"),
    (Join-Path $InstallRoot "app\config\local.json"),
    (Join-Path $InstallRoot "config\local.json"),
    (Join-Path $InstallRoot "local.json")
)
$PythonCandidates = @(
    $env:LOWVRAM3D_CONTROL_PYTHON,
    (Join-Path $InstallRoot "envs\control\Scripts\python.exe")
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
    return (
        (Test-Path -LiteralPath (Join-Path $CandidateDir "proof\job_receipt.json") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir "uv\game_ready_uv.glb") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir "views\mv_adapter\front.png") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir "views\mv_adapter\right.png") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir "views\mv_adapter\back.png") -PathType Leaf) -and
        (Test-Path -LiteralPath (Join-Path $CandidateDir "views\mv_adapter\left.png") -PathType Leaf)
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
        throw "No existing job contains a receipt, game_ready_uv.glb and four MV-Adapter views."
    }
    $JobId = $candidate.Name
}

$JobDir = Join-Path $JobsRoot $JobId
if (-not (Test-BaselineJob $JobDir)) {
    throw "Job $JobId is not ready for textured-baseline recovery: $JobDir"
}

$python = Resolve-ExistingFile $PythonCandidates "LowVRAM3D control Python"
$config = Resolve-ExistingFile $ConfigCandidates "LowVRAM3D config/local.json"

Write-Host "Recovering textured baseline from job $JobId" -ForegroundColor Cyan
Write-Host "UV mesh: $(Join-Path $JobDir 'uv\game_ready_uv.glb')"
Write-Host "Views:   $(Join-Path $JobDir 'views\mv_adapter')"

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

$env:PYTHONPATH = "$RepoRoot\src;$RepoRoot"
& $python (Join-Path $RepoRoot "workers\recover_textured_baseline.py") `
    --config $config `
    --job-id $JobId `
    --output-dir $ArtifactDir `
    --atlas-size $AtlasSize
$exitCode = $LASTEXITCODE

if ($exitCode -ne 0) {
    throw "Textured-baseline recovery failed with exit code $exitCode"
}

$glb = Join-Path $ArtifactDir "shaman_textured_baseline.glb"
$result = Join-Path $ArtifactDir "baseline_result.json"
if (-not (Test-Path -LiteralPath $glb -PathType Leaf)) {
    throw "Recovery returned success without the textured GLB: $glb"
}
if ((Get-Item -LiteralPath $glb).Length -le 0) {
    throw "Recovered textured GLB is empty: $glb"
}
if (-not (Test-Path -LiteralPath $result -PathType Leaf)) {
    throw "Recovery returned success without baseline_result.json: $result"
}

Set-Content -LiteralPath (Join-Path $ArtifactDir "source_job_id.txt") -Value $JobId -Encoding utf8
Write-Host "SHAMAN_TEXTURED_BASELINE_READY job=$JobId glb=$glb" -ForegroundColor Green
