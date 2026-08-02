<#
.SYNOPSIS
    One corrected SD2.1 I2MV canary with the condition encoder running in FP32.

.DESCRIPTION
    The numeric-boundary diagnostic proved the first nonfinite tensor is
    adapter_state_00 (CONTROL_ENCODER_OUTPUT): a fully finite FP16 control tensor
    in [0,1] enters the T2IAdapter condition encoder and zero finite values come
    out. This canary changes precision at that one boundary only - the condition
    encoder runs in FP32 and its residuals are validated before being cast back
    to the UNet latent dtype.

    Unchanged: control normalisation (official do_normalize=False), the UNet, the
    FP32 VAE boundaries, the reference-cache relay, the scheduler, the weights,
    seed 42, the six-view camera order and the prompts.

    Performs no texture projection and no candidate promotion. Rejects black,
    duplicate or nonfinite output.
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
    [string]$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\mv-adapter-cond-fp32-canary"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Worker = Join-Path $RepoRoot "workers\run_mv_adapter_cond_fp32_canary.py"

# Every rejected black-output run stays immutable evidence.
$ProtectedOutputDirs = @(
    "C:\AI\LowVRAM3D-benchmarks\mv-adapter-fp32-vae-canary\canary-20260802-111500",
    "C:\AI\LowVRAM3D-benchmarks\mv-adapter-fp32-vae-canary\canary-20260802-111851",
    "C:\AI\LowVRAM3D-benchmarks\mv-adapter-numeric-boundary-diagnostic"
)

foreach ($pair in @(
    @{ Label = "canary worker"; Path = $Worker },
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
$runRoot = Join-Path $OutputRoot "cond-fp32-$stamp"
foreach ($protectedDir in $ProtectedOutputDirs) {
    if ($runRoot -like "$protectedDir*") {
        throw "refusing to write into protected evidence directory: $protectedDir"
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
Write-Host "PROTECTED_OUTPUT_DIRS=$($ProtectedOutputDirs -join ';')"
Write-Host "CORRECTION=condition_encoder_fp32_only"
Write-Host "SETTINGS=320x320 steps=4 seed=42 views=6"

$logPath = Join-Path $runRoot "cond-fp32-canary.log"
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
        --expected-source-sha256 $ExpectedSourceSha256 `
        --resolution 320 `
        --steps 4 `
        --seed 42 2>&1 | Tee-Object -FilePath $logPath
    $workerExit = $LASTEXITCODE
}
finally {
    $ErrorActionPreference = $oldErrorActionPreference
}

$reportPath = Join-Path $runRoot "mv_adapter_cond_fp32_canary_report.json"
if (-not (Test-Path -LiteralPath $reportPath -PathType Leaf)) {
    throw "canary produced no report at $reportPath (exit $workerExit)"
}

$report = Get-Content -LiteralPath $reportPath -Raw | ConvertFrom-Json

$firstState = $null
if ($report.PSObject.Properties.Name -contains "condition_encoder_runtime" -and
    $report.condition_encoder_runtime -and
    $report.condition_encoder_runtime.adapter_state_records) {
    $firstState = $report.condition_encoder_runtime.adapter_state_records[0]
}

Write-Host ""
Write-Host "=========== CONDITION ENCODER FP32 CANARY VERDICT ==========="
Write-Host "STATUS=$($report.status)"
Write-Host "PASSED=$($report.passed)"
if ($report.PSObject.Properties.Name -contains "condition_encoder_checkpoint_coverage") {
    $coverage = $report.condition_encoder_checkpoint_coverage
    Write-Host "COND_ENCODER_CHECKPOINT_COVERAGE=$($coverage.matched_key_count)/$($coverage.condition_encoder_key_count)"
}
Write-Host "COND_ENCODER_DTYPE=$($report.condition_encoder_dtype_after_offload)"
Write-Host "COND_ENCODER_BOUNDARY=$($report.condition_encoder_boundary_placement)"
if ($firstState) {
    $s = $firstState.statistics
    Write-Host "ADAPTER_STATE_00 finite=$($s.finite_count) nonfinite=$($s.nonfinite_count) min=$($s.minimum) max=$($s.maximum) mean=$($s.mean) std=$($s.standard_deviation) abs_max=$($s.absolute_maximum)"
} else {
    Write-Host "ADAPTER_STATE_00=not_recorded"
}
Write-Host "FAILURE_CODES=$($report.failure_codes -join ',')"
Write-Host "PEAK_CUDA_MEMORY_BYTES=$($report.peak_cuda_memory_bytes)"
Write-Host "TEXTURE_PROJECTION_STARTED=$($report.texture_projection_started)"
Write-Host "CANDIDATE_PROMOTED=$($report.candidate_promoted)"
Write-Host "CONTACT_SHEET=$($report.contact_sheet)"
Write-Host "REPORT=$reportPath"
Write-Host "LOG=$logPath"
Write-Host "============================================================"

if ($workerExit -ne 0) {
    throw "corrected condition-encoder FP32 canary was rejected (exit $workerExit, status $($report.status))"
}
Write-Host "OPEN_THE_CONTACT_SHEET_AND_CLASSIFY_ANGLE_DIVERSITY_BEFORE_ANY_PROJECTION"
exit 0
