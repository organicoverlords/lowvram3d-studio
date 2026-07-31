<#
.SYNOPSIS
    One geometry-only Mini Turbo iteration for the antlered bird-shaman anchor.

.DESCRIPTION
    Mini Turbo is the only permitted generator. There is no TripoSR, SF3D or proxy_generate
    fallback anywhere in this script: if Mini Turbo fails, the run stops with the generator's own
    error. The raw GLB is preserved byte-for-byte, then validated in a fresh Blender process
    (import, finite bounds, triangle/vertex counts, orientation, round-trip, four neutral renders).

    Nothing here textures, UV unwraps, bakes, decimates, splits, poses, rigs or animates.
#>
[CmdletBinding()]
param(
    [string]$ImagePath = "C:\Users\Lauri\Downloads\ChatGPT Image 29.7.2026 klo 20.00.45.png",
    [string]$ExpectedSha256 = "4d23adc758c5b700dd29939e37c043ce61919792b566bdcf13f58b1409d6cf6f",
    [string]$MiniTurboPython = "C:\AI\HY3D2\python_standalone\python.exe",
    [string]$Hunyuan3DRoot = "C:\AI\HY3D2\Hunyuan3D-2",
    [string]$ModelRoot = "C:\AI\HY3D2\HuggingFaceHub\hunyuan3d-2mini-direct",
    [string]$Subfolder = "hunyuan3d-dit-v2-mini-turbo",
    [string]$BlenderPath = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe",
    [string]$OctreeLadder = "384:3000,320:2000,256:1500",
    [int]$Steps = 5
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$CanonicalRoot = "C:\AI\LowVRAM3D-benchmarks"
$RunStamp = Get-Date -Format "yyyyMMdd-HHmmss"
$OutputDir = Join-Path $CanonicalRoot "outputs\antlered_bird_shaman_anchor\mini-turbo-iterations\$RunStamp"
$LatestDir = Join-Path $CanonicalRoot "outputs\antlered_bird_shaman_anchor\geometry-latest"

function Assert-File([string]$Path, [string]$Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf) -or (Get-Item -LiteralPath $Path).Length -le 0) {
        throw "Missing or empty ${Label}: $Path"
    }
}

# ---------------------------------------------------------------- setup diagnostics
Write-Host "== Mini Turbo setup diagnostics ==" -ForegroundColor Cyan
Assert-File $ImagePath "shaman source image"
$actualHash = (Get-FileHash -LiteralPath $ImagePath -Algorithm SHA256).Hash.ToLowerInvariant()
if ($actualHash -ne $ExpectedSha256.ToLowerInvariant()) {
    throw "Shaman image SHA-256 mismatch. Expected $ExpectedSha256, got $actualHash"
}
Write-Host "source sha256 verified: $actualHash"

Assert-File $MiniTurboPython "Mini Turbo Python"
Assert-File (Join-Path $ModelRoot "$Subfolder\model.fp16.safetensors") "Mini Turbo DiT weights"
Assert-File $BlenderPath "Blender executable"
if (-not (Test-Path -LiteralPath $Hunyuan3DRoot -PathType Container)) {
    throw "Hunyuan3D root missing: $Hunyuan3DRoot"
}
Write-Host "mini turbo weights + blender present"

New-Item -ItemType Directory -Path $OutputDir -Force | Out-Null
$rawGlb = Join-Path $OutputDir "shaman_geometry_master.glb"
$runResult = Join-Path $OutputDir "mini_turbo_run_result.json"

$prompt = @"
Antlered bird-shaman character, upright full body, clear avian head and beak, broad coherent
branching antlers, layered feather and robe forms, readable shoulders, torso, arms and legs, and a
complete staff held separately from the body. Preserve the source silhouette and major proportions.
No horizontal collapse, compressed body, merged staff, duplicate limbs, missing antlers, floating
debris, or missing major silhouette features.
"@ -replace "\s+", " "

# ---------------------------------------------------------------- matte
# rembg/u2net erases every ornament hanging from the antler pole, erases the cords, and turns the
# staff ring into a black blob; those failures reappear in the geometry as missing silhouette
# features and floating debris. Key the flat studio plate instead.
Write-Host "== Matte ==" -ForegroundColor Cyan
$env:PYTHONPATH = $Hunyuan3DRoot
$env:PYTHONUNBUFFERED = "1"
$matte = Join-Path $OutputDir "shaman_matte.png"
& $MiniTurboPython (Join-Path $RepoRoot "workers\shaman_matte.py") `
    --image $ImagePath `
    --output $matte `
    --mode hybrid `
    --tolerance 60 `
    --enclosed-tolerance 32 `
    --enclosed-min-area 5000 `
    --shadow-tolerance 155 `
    --shadow-from 0.78 `
    --preview (Join-Path $OutputDir "shaman_matte_preview.png") `
    --stats-json (Join-Path $OutputDir "shaman_matte_stats.json")
if ($LASTEXITCODE -ne 0) { throw "Matte generation failed" }
Assert-File $matte "matted conditioning image"

# ---------------------------------------------------------------- generation (Mini Turbo only)
Write-Host "== Mini Turbo generation ==" -ForegroundColor Cyan
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
$env:HF_HUB_OFFLINE = "1"

& $MiniTurboPython (Join-Path $RepoRoot "workers\mini_turbo_generate.py") `
    --image $ImagePath `
    --conditioning-image $matte `
    --expected-image-sha256 $ExpectedSha256 `
    --output $rawGlb `
    --result-json $runResult `
    --prompt $prompt `
    --model-root $ModelRoot `
    --subfolder $Subfolder `
    --steps $Steps `
    --octree-ladder $OctreeLadder
if ($LASTEXITCODE -ne 0) {
    throw "Mini Turbo generation failed. No fallback generator is permitted. See $runResult"
}
Assert-File $rawGlb "raw Mini Turbo GLB"
$rawHash = (Get-FileHash -LiteralPath $rawGlb -Algorithm SHA256).Hash.ToLowerInvariant()
Write-Host "raw master sha256: $rawHash"

# ---------------------------------------------------------------- fresh Blender validation
Write-Host "== Fresh Blender validation ==" -ForegroundColor Cyan
$roundtrip = Join-Path $OutputDir "shaman_geometry_roundtrip.glb"
$validation = Join-Path $OutputDir "geometry_validation.json"
$previewDir = Join-Path $OutputDir "previews"

$env:PYTHONPATH = @((Join-Path $RepoRoot "blender"), (Join-Path $RepoRoot "src")) -join [IO.Path]::PathSeparator
& $BlenderPath --background --python-use-system-env `
    --python (Join-Path $RepoRoot "blender\geometry_iteration_validate.py") -- `
    --input $rawGlb `
    --roundtrip-output $roundtrip `
    --validation $validation `
    --preview-dir $previewDir
if ($LASTEXITCODE -ne 0) {
    throw "Fresh Blender geometry validation failed for $rawGlb"
}
Assert-File $roundtrip "round-trip GLB"
Assert-File $validation "geometry validation report"

# The raw master must be unchanged by validation.
$postHash = (Get-FileHash -LiteralPath $rawGlb -Algorithm SHA256).Hash.ToLowerInvariant()
if ($postHash -ne $rawHash) {
    throw "Raw Mini Turbo master was modified during validation ($rawHash -> $postHash)"
}

# ---------------------------------------------------------------- publish geometry-latest
Write-Host "== Publishing geometry-latest ==" -ForegroundColor Cyan
if (Test-Path -LiteralPath $LatestDir) { Remove-Item -LiteralPath $LatestDir -Recurse -Force }
New-Item -ItemType Directory -Path $LatestDir -Force | Out-Null

Copy-Item -LiteralPath $rawGlb -Destination (Join-Path $LatestDir "shaman_geometry_master.glb") -Force
Copy-Item -LiteralPath $roundtrip -Destination (Join-Path $LatestDir "shaman_geometry_roundtrip.glb") -Force
Copy-Item -LiteralPath $validation -Destination (Join-Path $LatestDir "geometry_validation.json") -Force
Copy-Item -LiteralPath $runResult -Destination (Join-Path $LatestDir "mini_turbo_run_result.json") -Force
foreach ($view in @("front", "three_quarter", "side", "back")) {
    $src = Join-Path $previewDir "$view.png"
    Assert-File $src "$view preview"
    Copy-Item -LiteralPath $src -Destination (Join-Path $LatestDir "preview_$view.png") -Force
}

# Geometry-only receipt. This lane deliberately has no texture, UV, rig or pose stages to report.
$validationData = Get-Content -LiteralPath $validation -Raw | ConvertFrom-Json
$runData = Get-Content -LiteralPath $runResult -Raw | ConvertFrom-Json
$receipt = [ordered]@{
    operation             = "mini_turbo_geometry_iteration"
    status                = "passed"
    run_stamp             = $RunStamp
    selected_generator    = "mini_turbo"
    generator_locked      = $true
    fallback_generators   = @()
    model_root            = $ModelRoot
    model_subfolder       = $Subfolder
    source_image          = $ImagePath
    source_image_sha256   = $actualHash
    prompt                = $prompt
    raw_glb               = (Join-Path $LatestDir "shaman_geometry_master.glb")
    raw_glb_sha256        = $rawHash
    raw_glb_bytes         = (Get-Item -LiteralPath $rawGlb).Length
    octree_resolution     = $runData.octree_resolution
    num_chunks            = $runData.num_chunks
    steps                 = $runData.steps
    seed                  = $runData.seed
    validation            = $validationData
    target_fbx_used       = $false
    texture_pipeline_run  = $false
    uv_pipeline_run       = $false
    rig_pipeline_run      = $false
    pose_pipeline_run     = $false
    scope                 = "geometry generation + fresh Blender validation only"
}
$receipt | ConvertTo-Json -Depth 12 | Set-Content -LiteralPath (Join-Path $LatestDir "job_receipt.json") -Encoding utf8
Copy-Item -LiteralPath (Join-Path $LatestDir "job_receipt.json") -Destination (Join-Path $OutputDir "job_receipt.json") -Force

Write-Host "MINI_TURBO_ITERATION_READY" -ForegroundColor Green
Write-Host "LATEST:     $LatestDir"
Write-Host "MASTER:     $rawGlb"
Write-Host "SHA256:     $rawHash"
Write-Host "VALIDATION: $validation"
