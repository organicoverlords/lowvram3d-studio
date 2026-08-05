<#
.SYNOPSIS
    Locate the first nonfinite tensor boundary in the proven failing SD2.1 I2MV run.

.DESCRIPTION
    Reproduces the exact configuration that produced six black views
    (320x320, 4 steps, seed 42, six views, DDPM shift-SNR scheduler, FP32 VAE
    boundaries, reference-cache relay, condition-encoder CPU offload) and records
    finite statistics at every tensor boundary from the reference latents to the
    decoded image. Fails closed at the first nonfinite tensor, after writing the
    report.

    Read-only with respect to model weights, adapter weights, the scheduler and
    the previous black-output directory. Performs no texture projection and no
    candidate promotion.
#>
[CmdletBinding()]
param(
    [string]$OfficialRepo = "C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\thirdparty\MV-Adapter",
    [string]$BaseModel = "C:\AI\HY3D2\HuggingFaceHub\models--stabilityai--stable-diffusion-2-1-base",
    [string]$AdapterFile = "C:\AI\HY3D2\HuggingFaceHub\models--huanngzh--mv-adapter\snapshots\6de4033df6b53366f3c009d22f5ec434bb55e59f\mvadapter_i2mv_sd21.safetensors",
    [string]$SourceImage = "C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png",
    [string]$ExpectedAdapterSha256 = "cf95469bc968efc5f042284a3043c66e940994d9f757a1a098b2403efb5d0016",
    [string]$ExpectedSourceSha256 = "4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f",
    [string]$ExpectedBranch = "magicmusic/parts-pose-materials-20260802",
    [string]$PythonExe = "C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\mv-adapter\Scripts\python.exe",
    [string]$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\mv-adapter-numeric-boundary-diagnostic"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Worker = Join-Path $RepoRoot "workers\diagnose_mv_adapter_numeric_boundary.py"

# Preserve every known rejected black-output run as immutable evidence. The
# proven run from the MagicMusic receipt is canary-20260802-111500. The second
# path is retained because an earlier handoff named it, but it may not exist.
$ProtectedBlackOutputDirs = @(
    "C:\AI\LowVRAM3D-benchmarks\mv-adapter-fp32-vae-canary\canary-20260802-111500",
    "C:\AI\LowVRAM3D-benchmarks\mv-adapter-fp32-vae-canary\canary-20260802-111851"
)

foreach ($pair in @(
    @{ Label = "diagnostic worker"; Path = $Worker },
    @{ Label = "python interpreter"; Path = $PythonExe },
    @{ Label = "adapter weight"; Path = $AdapterFile },
    @{ Label = "source image"; Path = $SourceImage }
)) {
    if (-not (Test-Path -LiteralPath $pair.Path -PathType Leaf)) {
        throw "$($pair.Label) is missing: $($pair.Path)"
    }
}
foreach ($pair in @(
    @{ Label = "official MV-Adapter repository"; Path = $OfficialRepo },
    @{ Label = "SD2.1 base model"; Path = $BaseModel }
)) {
    if (-not (Test-Path -LiteralPath $pair.Path -PathType Container)) {
        throw "$($pair.Label) directory is missing: $($pair.Path)"
    }
}

$origin = (& git -C $RepoRoot remote get-url origin).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "failed to read repository origin"
}
$branch = (& git -C $RepoRoot branch --show-current).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "failed to read repository branch"
}
$localHead = (& git -C $RepoRoot rev-parse HEAD).Trim()
if ($LASTEXITCODE -ne 0) {
    throw "failed to read repository HEAD"
}
$dirty = @(& git -C $RepoRoot status --porcelain)
if ($LASTEXITCODE -ne 0) {
    throw "failed to read repository status"
}

if ($origin -notmatch "organicoverlords/lowvram3d-studio(?:\.git)?$") {
    throw "repository identity mismatch: $origin"
}
if ($branch -ne $ExpectedBranch) {
    throw "branch mismatch: expected=$ExpectedBranch actual=$branch"
}
if (-not [string]::IsNullOrWhiteSpace(($dirty -join "`n"))) {
    throw "feature worktree is dirty"
}

& git -C $RepoRoot fetch --prune origin
if ($LASTEXITCODE -ne 0) {
    throw "git fetch failed"
}
$remoteHead = (& git -C $RepoRoot rev-parse "origin/$ExpectedBranch").Trim()
if ($LASTEXITCODE -ne 0) {
    throw "failed to resolve remote feature HEAD"
}
if ($localHead -ne $remoteHead) {
    throw "local and remote feature HEAD differ: local=$localHead remote=$remoteHead"
}

$actualAdapterSha256 = (Get-FileHash -LiteralPath $AdapterFile -Algorithm SHA256).Hash.ToLowerInvariant()
$actualSourceSha256 = (Get-FileHash -LiteralPath $SourceImage -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualAdapterSha256 -ne $ExpectedAdapterSha256.ToLowerInvariant()) {
    throw "adapter hash mismatch: expected=$ExpectedAdapterSha256 actual=$actualAdapterSha256"
}
if ($actualSourceSha256 -ne $ExpectedSourceSha256.ToLowerInvariant()) {
    throw "source-image hash mismatch: expected=$ExpectedSourceSha256 actual=$actualSourceSha256"
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $OutputRoot "numeric-boundary-$stamp"
foreach ($protectedDir in $ProtectedBlackOutputDirs) {
    if ($runRoot -like "$protectedDir*") {
        throw "refusing to write into protected black-output directory: $protectedDir"
    }
}
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

Write-Host "ORIGIN=$origin"
Write-Host "BRANCH=$branch"
Write-Host "HEAD=$localHead"
Write-Host "REMOTE_HEAD=$remoteHead"
Write-Host "WORKTREE_CLEAN=true"
Write-Host "SOURCE_SHA256=$actualSourceSha256"
Write-Host "ADAPTER_SHA256=$actualAdapterSha256"
Write-Host "REPO_ROOT=$RepoRoot"
Write-Host "OUTPUT_ROOT=$runRoot"
Write-Host "PROTECTED_BLACK_OUTPUT_DIRS=$($ProtectedBlackOutputDirs -join ';')"
Write-Host "SETTINGS=320x320 steps=4 seed=42 views=6"

$logPath = Join-Path $runRoot "numeric-boundary-diagnostic.log"
$oldErrorActionPreference = $ErrorActionPreference
try {
    # Python progress bars write to stderr during healthy execution. Keep them in
    # the durable log without letting Windows PowerShell promote them to a
    # terminating NativeCommandError.
    $ErrorActionPreference = "Continue"
    & $PythonExe $Worker `
        --official-repo $OfficialRepo `
        --base-model $BaseModel `
        --adapter-file $AdapterFile `
        --source-image $SourceImage `
        --output-root $runRoot `
        --expected-adapter-sha256 $ExpectedAdapterSha256 `
        --resolution 320 `
        --steps 4 `
        --seed 42 2>&1 | Tee-Object -FilePath $logPath
    $workerExit = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $oldErrorActionPreference
}

$reportPath = Join-Path $runRoot "mv_adapter_numeric_boundary_report.json"
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    throw "diagnostic produced no report at $reportPath (exit $workerExit)"
}

$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json

Write-Host ""
Write-Host "================ NUMERIC BOUNDARY VERDICT ================"
Write-Host "STATUS=$($report.status)"
Write-Host "DIAGNOSTIC_COMPLETE=$($report.diagnostic_complete)"
Write-Host "NONFINITE_BOUNDARY_FOUND=$($report.nonfinite_boundary_found)"
Write-Host "FIRST_NONFINITE_LABEL=$($report.first_nonfinite_label)"
Write-Host "BOUNDARY_CATEGORY=$($report.boundary_category)"
Write-Host "DECISION=$($report.decision)"
Write-Host "DECISION_FROM_TASK_RULES=$($report.decision_from_task_rules)"
Write-Host "PROBE_RECORD_COUNT=$($report.probe_record_count)"
Write-Host "PEAK_CUDA_MEMORY_BYTES=$($report.peak_cuda_memory_bytes)"
Write-Host "TEXTURE_PROJECTION_STARTED=$($report.texture_projection_started)"
Write-Host "CANDIDATE_PROMOTED=$($report.candidate_promoted)"
Write-Host "REPORT=$reportPath"
Write-Host "LOG=$logPath"
Write-Host "=========================================================="

if ($workerExit -ne 0) {
    throw "numeric boundary diagnostic failed without a verdict (exit $workerExit)"
}
exit 0
