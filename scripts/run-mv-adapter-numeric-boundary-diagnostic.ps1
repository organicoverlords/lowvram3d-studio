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
    [string]$PythonExe = "C:\Users\Lauri\AppData\Local\LowVRAM3DStudio\envs\mv-adapter\Scripts\python.exe",
    [string]$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\mv-adapter-numeric-boundary-diagnostic"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$Worker = Join-Path $RepoRoot "workers\diagnose_mv_adapter_numeric_boundary.py"

# The previous black-output run must stay byte-identical as evidence.
$ProtectedBlackOutputDir = "C:\AI\LowVRAM3D-benchmarks\mv-adapter-fp32-vae-canary\canary-20260802-111851"

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

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$runRoot = Join-Path $OutputRoot "numeric-boundary-$stamp"
if ($runRoot -like "$ProtectedBlackOutputDir*") {
    throw "refusing to write into the protected black-output directory: $ProtectedBlackOutputDir"
}
New-Item -ItemType Directory -Force -Path $runRoot | Out-Null

Write-Host "REPO_ROOT=$RepoRoot"
Write-Host "OUTPUT_ROOT=$runRoot"
Write-Host "PROTECTED_BLACK_OUTPUT_DIR=$ProtectedBlackOutputDir"
Write-Host "SETTINGS=320x320 steps=4 seed=42 views=6"

$logPath = Join-Path $runRoot "numeric-boundary-diagnostic.log"

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
$reportPath = Join-Path $runRoot "mv_adapter_numeric_boundary_report.json"

if (-not (Test-Path -LiteralPath $reportPath)) {
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
Write-Host "REPORT=$reportPath"
Write-Host "LOG=$logPath"
Write-Host "=========================================================="

if ($workerExit -ne 0) {
    throw "numeric boundary diagnostic failed without a verdict (exit $workerExit)"
}
exit 0
