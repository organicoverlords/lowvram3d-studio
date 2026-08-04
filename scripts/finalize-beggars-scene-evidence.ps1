[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

$expectedRepository = 'organicoverlords/lowvram3d-studio'
$expectedBranch = 'agent/blender-beggars-scene-20260804'
if ($env:GITHUB_REPOSITORY -ne $expectedRepository) {
    throw "Repository mismatch: $env:GITHUB_REPOSITORY"
}
if ($env:GITHUB_REF_NAME -ne $expectedBranch) {
    throw "Branch mismatch: $env:GITHUB_REF_NAME"
}
$remote = (& git remote get-url origin | Out-String).Trim()
if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
    throw "Remote mismatch: $remote"
}

$artifactRoot = $env:BEGGARS_V2_OUTPUT
if (-not $artifactRoot) {
    $artifactRoot = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\run-$env:GITHUB_RUN_ID\artifact"
}
if (-not (Test-Path -LiteralPath $artifactRoot)) {
    throw "Artifact root is missing: $artifactRoot"
}

$requiredFiles = @(
    'beggars_photoreal_recreation.blend',
    'beggars_photoreal_recreation.mp4',
    'hero_clean_render.png',
    'wide_scene_proof.png',
    'contact_sheet.png',
    'scene_receipt.json',
    'artifact_manifest.json',
    'reference_reconstruction_report.json',
    'worker_receipt.json'
)
foreach ($name in $requiredFiles) {
    $path = Join-Path $artifactRoot $name
    if (-not (Test-Path -LiteralPath $path)) {
        throw "Required artifact is missing: $name"
    }
    if ((Get-Item -LiteralPath $path).Length -le 0) {
        throw "Required artifact is empty: $name"
    }
}
if ((Get-Item -LiteralPath (Join-Path $artifactRoot 'beggars_photoreal_recreation.blend')).Length -lt 1000000) {
    throw 'Blender artifact is implausibly small.'
}
if ((Get-Item -LiteralPath (Join-Path $artifactRoot 'beggars_photoreal_recreation.mp4')).Length -lt 100000) {
    throw 'Final MP4 is implausibly small.'
}

$scene = Get-Content -LiteralPath (Join-Path $artifactRoot 'scene_receipt.json') -Raw | ConvertFrom-Json
$manifest = Get-Content -LiteralPath (Join-Path $artifactRoot 'artifact_manifest.json') -Raw | ConvertFrom-Json
$worker = Get-Content -LiteralPath (Join-Path $artifactRoot 'worker_receipt.json') -Raw | ConvertFrom-Json
$reconstruction = Get-Content -LiteralPath (Join-Path $artifactRoot 'reference_reconstruction_report.json') -Raw | ConvertFrom-Json

if ($scene.classification -ne 'USER_VISUAL_REVIEW_REQUIRED') {
    throw "Unexpected scene classification: $($scene.classification)"
}
if (-not $scene.reload_validation.blend_reloaded -or @($scene.reload_validation.missing_required_objects).Count -ne 0) {
    throw 'Blender save/reload validation is not proven.'
}
if ([int]$scene.tracked_input_frames -lt 10 -or [int]$scene.face_vertices -lt 10000) {
    throw 'Tracked 3D face evidence is insufficient.'
}
if ($manifest.automated_build_and_validation -ne 'PROVEN') {
    throw "Automated artifact validation is not proven: $($manifest.automated_build_and_validation)"
}
if ($worker.classification -ne 'PROVEN') {
    throw "Worker receipt is not proven: $($worker.classification)"
}
if ($reconstruction.classification -ne 'PROVEN') {
    throw "Face reconstruction is not proven: $($reconstruction.classification)"
}
if ($manifest.reference_media_in_artifact -ne $false) {
    throw 'Reference media was unexpectedly packaged.'
}

$evidence = Join-Path $PWD 'evidence\latest-beggars-scene'
New-Item -ItemType Directory -Path $evidence -Force | Out-Null
foreach ($name in @('scene_receipt.json','artifact_manifest.json','reference_reconstruction_report.json','worker_receipt.json')) {
    Copy-Item -LiteralPath (Join-Path $artifactRoot $name) -Destination (Join-Path $evidence $name) -Force
}

$runtimePatchedSources = @(
    'blender/build_beggars_meme_scene.py',
    'tools/beggars_scene/prepare_reference_sequence.py'
)
foreach ($source in $runtimePatchedSources) {
    & git update-index --no-assume-unchanged -- $source 2>$null
    & git checkout -- $source 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Could not restore runtime-patched source: $source"
    }
}
if (@(& git status --porcelain -- $runtimePatchedSources).Count -gt 0) {
    throw 'Runtime-patched Python sources remain dirty after restoration.'
}

git config user.name 'github-actions[bot]'
git config user.email '41898282+github-actions[bot]@users.noreply.github.com'
git add -- evidence/latest-beggars-scene
if (@(& git status --porcelain -- evidence/latest-beggars-scene).Count -gt 0) {
    & git commit -m "evidence(scene): record beggars visual-repair run $env:GITHUB_RUN_ID"
    if ($LASTEXITCODE -ne 0) {
        throw 'Could not commit compact beggars evidence.'
    }
}

& git fetch origin $expectedBranch
if ($LASTEXITCODE -ne 0) {
    throw 'Could not fetch the latest dedicated branch before evidence publication.'
}
& git rebase "origin/$expectedBranch"
if ($LASTEXITCODE -ne 0) {
    & git rebase --abort 2>$null
    throw 'Could not rebase compact evidence over the latest run-registration commit.'
}
& git push origin "HEAD:$expectedBranch"
if ($LASTEXITCODE -ne 0) {
    throw 'Could not publish compact evidence after race-safe rebase.'
}

Write-Host 'BEGGARS_AUTOMATED_BUILD=PROVEN'
Write-Host 'BEGGARS_RUNTIME_PATCH_RESTORE=PROVEN'
Write-Host 'BEGGARS_EVIDENCE_PUSH_RACE_REPAIRED=PROVEN'
Write-Host 'BEGGARS_VISUAL_MATCH=USER_REVIEW_REQUIRED'
