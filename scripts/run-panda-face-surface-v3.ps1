param(
    [string]$RunId = "local-$(Get-Date -Format yyyyMMdd-HHmmss)",
    [string]$OutputRoot = "C:\AI\LowVRAM3D-benchmarks\production\panda_face_minimal_surface_20260806"
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = "C:\AI\HY3D2\python_standalone\python.exe"
$blender = "C:\Program Files\Blender Foundation\Blender 5.2\blender.exe"
$baselineGlb = "C:\AI\panda_support_local_2048\candidate_2048\panda_atlas_support_fixed_2048.glb"
$baselineAtlas = "C:\AI\panda_support_local_2048\candidate_2048\atlas_2048_nearest.png"
$sourceImage = "C:\AI\LowVRAM3D-benchmarks\images\red_panda_character.png"
$fixture = Join-Path $repo "fixtures\panda_face_source_fixture_20260806.json"
$runRoot = Join-Path $OutputRoot $RunId
$candidate = Join-Path $runRoot "candidate_a"
$renders = Join-Path $runRoot "renders"
$baselineRender = Join-Path $renders "baseline"
$candidateRender = Join-Path $renders "candidate_a"

New-Item -ItemType Directory -Force $candidate,$baselineRender,$candidateRender | Out-Null
$env:PYTHONPATH = "$repo\workers;$repo"

& $python -m pytest -q -p no:cacheprovider `
    (Join-Path $repo "tests\test_face_surface_candidate_v3.py") `
    (Join-Path $repo "tests\test_face_surface_candidate_v2.py") `
    (Join-Path $repo "tests\test_face_surface_ownership_core.py") `
    (Join-Path $repo "tests\test_face_patch_texture_contract.py")
if ($LASTEXITCODE -ne 0) { throw "Focused tests failed: $LASTEXITCODE" }

& $python (Join-Path $repo "workers\face_surface_candidate_v3.py") `
    --baseline-glb $baselineGlb --baseline-atlas $baselineAtlas `
    --source-image $sourceImage --source-fixture $fixture `
    --output-dir $candidate --ray-stride 3 --grow-target 350 --build-textured
if ($LASTEXITCODE -ne 0) { throw "Candidate build failed: $LASTEXITCODE" }

& $blender --background --factory-startup --python (Join-Path $repo "blender\render_glb_diagnostic_set.py") -- `
    --input $baselineGlb --out-dir $baselineRender --label baseline --resolution 768
if ($LASTEXITCODE -ne 0) { throw "Baseline render failed: $LASTEXITCODE" }
& $blender --background --factory-startup --python (Join-Path $repo "blender\render_glb_diagnostic_set.py") -- `
    --input (Join-Path $candidate "panda_face_surface_owned_2048.glb") `
    --out-dir $candidateRender --label candidate_a --resolution 768
if ($LASTEXITCODE -ne 0) { throw "Candidate render failed: $LASTEXITCODE" }

& $python (Join-Path $repo "workers\build_contact_sheet.py") --render-dir $baselineRender --output (Join-Path $baselineRender "contact_sheet.png")
& $python (Join-Path $repo "workers\build_contact_sheet.py") --render-dir $candidateRender --output (Join-Path $candidateRender "contact_sheet.png")
Write-Output "v3 evidence written to $runRoot"
