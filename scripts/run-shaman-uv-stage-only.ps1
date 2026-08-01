[CmdletBinding()]
param(
    [string]$BenchmarkRoot = "C:\AI\LowVRAM3D-benchmarks",
    [string]$FinalPipelineRoot = "",
    [string]$BlenderPath = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    [int]$Resolution = 4096,
    [double]$MaxOverlapTexels = 1.0,
    [double]$OverlapTimeoutSeconds = 120.0,
    [int]$MaxCandidatePairs = 10000000
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$UvScript = Join-Path $RepoRoot "blender\final_pipeline_uv.py"

if (-not (Test-Path -LiteralPath $BlenderPath)) {
    $candidate = Get-Command blender.exe -ErrorAction SilentlyContinue
    if ($candidate) {
        $BlenderPath = $candidate.Source
    } else {
        throw "Blender was not found at '$BlenderPath' or on PATH."
    }
}
if (-not (Test-Path -LiteralPath $UvScript)) {
    throw "UV worker is missing: $UvScript"
}

if (-not $FinalPipelineRoot) {
    $expected = Join-Path $BenchmarkRoot "outputs\antlered_bird_shaman_anchor\final-pipeline"
    if (Test-Path -LiteralPath (Join-Path $expected "game\shaman_lod0.glb")) {
        $FinalPipelineRoot = $expected
    } else {
        $found = Get-ChildItem -LiteralPath $BenchmarkRoot -Filter "shaman_lod0.glb" -File -Recurse -ErrorAction SilentlyContinue |
            Where-Object { $_.FullName -match '[\\/]final-pipeline[\\/]game[\\/]shaman_lod0\.glb$' } |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1
        if (-not $found) {
            throw "Could not find final-pipeline\game\shaman_lod0.glb under $BenchmarkRoot"
        }
        $FinalPipelineRoot = Split-Path (Split-Path $found.FullName -Parent) -Parent
    }
}

$FinalPipelineRoot = (Resolve-Path -LiteralPath $FinalPipelineRoot).Path
$GameDir = Join-Path $FinalPipelineRoot "game"
$ReportsDir = Join-Path $FinalPipelineRoot "reports"
$Input = Join-Path $GameDir "shaman_lod0.glb"
$Candidate = Join-Path $GameDir "shaman_lod0_uv_exact.glb"
$Canonical = Join-Path $GameDir "shaman_lod0_uv.glb"
$Report = Join-Path $ReportsDir "uv_quality_exact.json"
$Receipt = Join-Path $ReportsDir "uv_stage_receipt.json"

if (-not (Test-Path -LiteralPath $Input)) {
    throw "LOD0 input is missing: $Input"
}
New-Item -ItemType Directory -Path $GameDir -Force | Out-Null
New-Item -ItemType Directory -Path $ReportsDir -Force | Out-Null
Remove-Item -LiteralPath $Candidate -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath $Report -Force -ErrorAction SilentlyContinue

$InputHashBefore = (Get-FileHash -LiteralPath $Input -Algorithm SHA256).Hash.ToLowerInvariant()
$InputBytesBefore = (Get-Item -LiteralPath $Input).Length
$StartedAt = (Get-Date).ToUniversalTime()

$env:PYTHONPATH = "$RepoRoot\blender;$RepoRoot\src;$RepoRoot"

Write-Host "SHAMAN_UV_STAGE_START" -ForegroundColor Cyan
Write-Host "input=$Input"
Write-Host "input_sha256=$InputHashBefore"
Write-Host "candidate=$Candidate"
Write-Host "report=$Report"
Write-Host "overlap_timeout_seconds=$OverlapTimeoutSeconds"

& $BlenderPath `
    --background `
    --python-use-system-env `
    --python $UvScript `
    -- `
    --input $Input `
    --output $Candidate `
    --report $Report `
    --resolution $Resolution `
    --max-overlap-texels $MaxOverlapTexels `
    --overlap-timeout-seconds $OverlapTimeoutSeconds `
    --max-candidate-pairs $MaxCandidatePairs
$BlenderExit = $LASTEXITCODE

$InputHashAfter = (Get-FileHash -LiteralPath $Input -Algorithm SHA256).Hash.ToLowerInvariant()
$InputBytesAfter = (Get-Item -LiteralPath $Input).Length
if ($InputHashAfter -ne $InputHashBefore -or $InputBytesAfter -ne $InputBytesBefore) {
    throw "LOD0 input changed during UV processing. Before=$InputHashBefore After=$InputHashAfter"
}
if (-not (Test-Path -LiteralPath $Report)) {
    throw "UV stage did not produce its report. Blender exit=$BlenderExit"
}

$Metrics = Get-Content -LiteralPath $Report -Raw | ConvertFrom-Json
$GatePassed = [bool]$Metrics.gate_passed
$CandidatePresent = Test-Path -LiteralPath $Candidate
$Promoted = $false
$Backup = $null

# Every one of these must fail the stage. A timed-out detector reports zeroes for overlap because
# it never tested a pair, and those zeroes previously read as a clean result.
$FailureReasons = @()
if ($BlenderExit -ne 0) { $FailureReasons += "blender exit $BlenderExit" }
if (-not $GatePassed) { $FailureReasons += "gate_passed=false" }
if (-not $CandidatePresent) { $FailureReasons += "candidate GLB missing" }
if ([bool]$Metrics.exact_overlap.timed_out) { $FailureReasons += "exact overlap timed out" }
if (-not [bool]$Metrics.exact_overlap.success) { $FailureReasons += "exact overlap did not succeed" }
if ([int]$Metrics.exact_overlap.tested_pair_count -le 0) {
    $FailureReasons += "exact overlap tested 0 pairs (candidate_pair_count=$($Metrics.exact_overlap.candidate_pair_count))"
}
if ([int]$Metrics.degenerate_uv_triangles -ne 0) {
    $FailureReasons += "$($Metrics.degenerate_uv_triangles) degenerate UV triangles"
}
if ([int]$Metrics.exact_overlap.out_of_bounds_triangle_count -ne 0) {
    $FailureReasons += "$($Metrics.exact_overlap.out_of_bounds_triangle_count) out-of-bounds UV triangles"
}
if ([double]$Metrics.positive_overlap_total_texels_equivalent -gt $MaxOverlapTexels) {
    $FailureReasons += "overlap $($Metrics.positive_overlap_total_texels_equivalent) texels exceeds $MaxOverlapTexels"
}

if ($FailureReasons.Count -eq 0) {
    if (Test-Path -LiteralPath $Canonical) {
        $stamp = Get-Date -Format "yyyyMMdd-HHmmss"
        $Backup = Join-Path $GameDir "shaman_lod0_uv.pre-exact-$stamp.glb"
        Copy-Item -LiteralPath $Canonical -Destination $Backup -Force
    }
    Copy-Item -LiteralPath $Candidate -Destination $Canonical -Force
    $Promoted = $true
}

$CandidateHash = if ($CandidatePresent) {
    (Get-FileHash -LiteralPath $Candidate -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }
$CanonicalHash = if (Test-Path -LiteralPath $Canonical) {
    (Get-FileHash -LiteralPath $Canonical -Algorithm SHA256).Hash.ToLowerInvariant()
} else { $null }

$ReceiptObject = [ordered]@{
    schema_version = 1
    stage = "shaman_uv_exact_stage_only"
    started_at = $StartedAt.ToString("o")
    finished_at = (Get-Date).ToUniversalTime().ToString("o")
    repo_head = "$(& git -C $RepoRoot rev-parse HEAD)".Trim()
    # A detached worktree - which is exactly how this stage is meant to be run - makes
    # `git branch --show-current` return nothing, and calling .Trim() on that null threw before the
    # receipt was ever written, losing the evidence for every run.
    repo_branch = $(
        $current = "$(& git -C $RepoRoot branch --show-current)".Trim()
        if ($current) { $current } else { "DETACHED" }
    )
    input = $Input
    input_sha256 = $InputHashBefore
    input_bytes = $InputBytesBefore
    input_preserved = ($InputHashAfter -eq $InputHashBefore -and $InputBytesAfter -eq $InputBytesBefore)
    candidate = $Candidate
    candidate_present = $CandidatePresent
    candidate_sha256 = $CandidateHash
    canonical = $Canonical
    canonical_sha256 = $CanonicalHash
    previous_canonical_backup = $Backup
    report = $Report
    blender_exit = $BlenderExit
    gate_passed = $GatePassed
    promoted = $Promoted
    overlap_timeout_seconds = $OverlapTimeoutSeconds
    max_candidate_pairs = $MaxCandidatePairs
    atlas_utilisation = $Metrics.atlas_utilisation
    exact_overlap_pairs = $Metrics.positive_overlap_pair_count
    exact_overlap_texels = $Metrics.positive_overlap_total_texels_equivalent
    degenerate_uv_triangles = $Metrics.degenerate_uv_triangles
    exact_candidate_pairs = $Metrics.exact_overlap.candidate_pair_count
    exact_tested_pairs = $Metrics.exact_overlap.tested_pair_count
    exact_timed_out = $Metrics.exact_overlap.timed_out
    exact_success = $Metrics.exact_overlap.success
    out_of_bounds_triangles = $Metrics.exact_overlap.out_of_bounds_triangle_count
    failure_reasons = @($FailureReasons)
    errors = @($Metrics.errors)
    warnings = @($Metrics.warnings)
}
$ReceiptObject | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $Receipt -Encoding UTF8

Write-Host ""
Write-Host "SHAMAN_UV_STAGE_RESULT"
Write-Host "blender_exit=$BlenderExit"
Write-Host "gate_passed=$GatePassed"
Write-Host "promoted=$Promoted"
Write-Host "atlas_utilisation_percent=$([math]::Round([double]$Metrics.atlas_utilisation * 100.0, 4))"
Write-Host "exact_overlap_pairs=$($Metrics.positive_overlap_pair_count)"
Write-Host "exact_overlap_texels=$($Metrics.positive_overlap_total_texels_equivalent)"
Write-Host "degenerate_uv_triangles=$($Metrics.degenerate_uv_triangles)"
Write-Host "receipt=$Receipt"

if (-not $Promoted) {
    Write-Host "failure_reasons=$($FailureReasons -join '; ')" -ForegroundColor Yellow
    Write-Error "UV stage did not pass and was not promoted. Geometry and previous canonical UV were preserved." -ErrorAction Continue
    # exit, not throw: a throw leaves $LASTEXITCODE at whatever Blender set, so a caller checking
    # `if ($LASTEXITCODE -ne 0)` sails past a failed stage.
    exit 1
}

Write-Host "SHAMAN_UV_STAGE_PASSED" -ForegroundColor Green
exit 0
