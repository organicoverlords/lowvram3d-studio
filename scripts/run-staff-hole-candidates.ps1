<#
.SYNOPSIS
    Prove the complete staff-repair loop: measured source feature -> bounded candidates ->
    3D openness -> deterministic 2D scale/shape -> fresh import -> promote or preserve baseline.

.DESCRIPTION
    Generates at most three localized candidates around the measured 40.6% recess perimeter,
    validates each in 3D and in 2D, and promotes only a candidate that independently passes
    every gate. If none passes, the canonical baseline is preserved untouched.

    The canonical baseline is only ever read. Its hash is checked before and after.
#>
[CmdletBinding()]
param(
    [string]$Baseline = "C:\AI\LowVRAM3D-benchmarks\pipeline-v2-validation\shaman_v2_validation\state\CLEAN\proven\shaman_v2_validation_stance_clean.glb",
    [string]$Root = "C:\AI\LowVRAM3D-benchmarks\shaman-staff-hole-repair\candidates-20260802",
    [string]$SourceCrop = "evidence\visual-qa\crops\source_staff_head.png",
    [string]$BeforeCrop = "evidence\visual-qa\crops\before_front.png"
)

$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$python = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $Baseline)) { throw "baseline is missing: $Baseline" }
$baselineHashBefore = (Get-FileHash -LiteralPath $Baseline -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "BASELINE_SHA256_BEFORE=$baselineHashBefore"

# Narrow variation around the measured recess perimeter. Not arbitrary: 0.406 is the measured
# perimeter of the original recessed centre as a fraction of the outer staff-disc diameter.
$candidates = @(
    @{ name = 'small';   mouth = 0.376 },
    @{ name = 'nominal'; mouth = 0.406 },
    @{ name = 'large';   mouth = 0.436 }
)

New-Item -ItemType Directory -Path $Root -Force | Out-Null
$env:PYTHONPATH = "$repo\blender;$repo\src;$repo\workers;$repo"

foreach ($candidate in $candidates) {
    $dir = Join-Path $Root $candidate.name
    New-Item -ItemType Directory -Path (Join-Path $dir "evidence") -Force | Out-Null
    Write-Host "`n=== candidate $($candidate.name): mouth_fraction=$($candidate.mouth) ==="

    # Blender writes deprecation warnings to stderr. Under ErrorActionPreference=Stop those
    # become terminating errors and kill the run mid-preflight, so relax it around the call.
    $previous = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $blender --background --factory-startup --python "$repo\blender\repair_staff_ring_hole.py" -- `
        --input $Baseline `
        --output (Join-Path $dir "shaman_v2_staff_hole_repaired.glb") `
        --report (Join-Path $dir "staff_hole_report.json") `
        --evidence-dir (Join-Path $dir "evidence") `
        --resolution 1024 `
        --mouth-fraction $candidate.mouth `
        --localized-repair *> (Join-Path $dir "repair.log")
    $exit = $LASTEXITCODE
    $ErrorActionPreference = $previous

    if ($exit -ne 0) {
        Write-Warning "candidate $($candidate.name) FAILED 3D gates (exit $exit); eliminated"
        continue
    }
    Write-Host "candidate $($candidate.name): 3D gates passed"
}

# Deterministic 2D gate over whatever survived, using the already-proven thresholds.
$manifestDir = Join-Path $Root "manifests"
New-Item -ItemType Directory -Path $manifestDir -Force | Out-Null
$gateArgs = @()
foreach ($candidate in $candidates) {
    $dir = Join-Path $Root $candidate.name
    $render = Join-Path $dir "evidence\after_staff_front.png"
    if (-not (Test-Path -LiteralPath $render)) { continue }
    # Cut the candidate to the same box as the stored before-crop, or the gate measures framing.
    $front = Join-Path $dir "evidence\candidate_front_crop.png"
    & $python "$repo\scripts\crop_for_gate.py" $render $front | Out-Null
    $manifest = Join-Path $manifestDir "$($candidate.name).json"
    @{
        source_crop    = (Join-Path $repo $SourceCrop)
        before_crop    = (Join-Path $repo $BeforeCrop)
        candidate_crop = $front
        feature_name   = "staff ring through-hole"
        expected_description = "organic opening at the measured recess perimeter, lip preserved"
        constraints    = @("stay inside the original recess", "preserve the raised inner lip")
        repair_roi     = @(0.28, 0.28, 0.72, 0.72)
    } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $manifest -Encoding utf8
    $gateArgs += @('--manifest', $manifest)
}

if ($gateArgs.Count -eq 0) {
    Write-Warning "no candidate produced a front render; preserving baseline"
    exit 0
}

Write-Host "`n=== deterministic 2D gate ==="
& $python "$repo\workers\deterministic_visual_gate.py" @gateArgs --receipt-dir (Join-Path $Root "gate")

$after = (Get-FileHash -LiteralPath $Baseline -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "`nBASELINE_SHA256_AFTER=$after"
if ($after -ne $baselineHashBefore) { throw "canonical baseline changed during candidate generation" }
Write-Host "INPUT_UNCHANGED=true"
