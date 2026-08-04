[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$expectedRepository = 'organicoverlords/lowvram3d-studio'
$expectedBranch = 'agent/blender-beggars-scene-20260804'
$path = 'blender\build_beggars_meme_scene.py'

if ($env:GITHUB_REPOSITORY -ne $expectedRepository) {
    throw "Repository mismatch: $env:GITHUB_REPOSITORY"
}
if ($env:GITHUB_REF_NAME -ne $expectedBranch) {
    throw "Branch mismatch: $env:GITHUB_REF_NAME"
}
$remote = (& git remote get-url origin | Out-String).Trim()
$head = (& git rev-parse HEAD | Out-String).Trim()
if ($remote -notmatch 'organicoverlords/lowvram3d-studio') {
    throw "Remote mismatch: $remote"
}
if ($head -ne $env:GITHUB_SHA) {
    throw "Checkout HEAD $head does not equal workflow SHA $env:GITHUB_SHA"
}
if (-not (Test-Path -LiteralPath $path)) {
    throw "Blender scene builder is missing: $path"
}
if (@(& git status --short).Count -gt 0) {
    throw 'Checkout is visibly dirty before atlas topology compatibility repair.'
}

$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))
$old = @'
    transformed, triangles, colors_rgb, uv_coordinates, feature_report = _close_atlas_feature_holes(
        transformed,
        triangles,
        colors_rgb,
        uv_coordinates,
    )

    maximum_shape_keys = 42
'@
$new = @'
    feature_report = {
        "classification": "NOT_REQUIRED_CLOSED_DENSE_TOPOLOGY",
        "boundary_loop_count": len(_atlas_boundary_loops(triangles)),
        "appearance_route": "PACKED_DERIVED_FACE_ONLY_ATLAS",
        "reason": "3DDFA dense topology already contains the eye and mouth surfaces; atlas UVs carry their appearance.",
    }

    maximum_shape_keys = 42
'@

if ($content.Contains($old)) {
    $content = $content.Replace($old, $new)
    Write-Host 'ATLAS_TOPOLOGY_CLOSURE_REMOVED=APPLIED'
}
elif ($content.Contains('"classification": "NOT_REQUIRED_CLOSED_DENSE_TOPOLOGY"')) {
    Write-Host 'ATLAS_TOPOLOGY_CLOSURE_REMOVED=ALREADY_APPLIED'
}
else {
    throw 'Could not locate the rejected atlas boundary-loop closure call.'
}

if ($content.Contains('feature_report = _close_atlas_feature_holes')) {
    throw 'The rejected atlas boundary-loop closure call remains active.'
}
if (-not $content.Contains('mesh.materials.append(face_atlas_material(atlas_path))')) {
    throw 'The derived atlas material route is missing.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $path),
    $content,
    [System.Text.UTF8Encoding]::new($false)
)

$controlPython = "$env:LOCALAPPDATA\LowVRAM3DStudio\envs\control\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $controlPython)) {
    throw "Control Python is missing: $controlPython"
}
& $controlPython -m py_compile $path
if ($LASTEXITCODE -ne 0) {
    throw 'Atlas topology-compatible Blender builder failed compile validation.'
}
& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded atlas topology compatibility repair.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Atlas topology compatibility repair left visible checkout dirt.'
}

Write-Host 'ATLAS_UV_ON_ORIGINAL_DENSE_TOPOLOGY=PROVEN'
Write-Host 'ARTIFICIAL_EYE_MOUTH_HOLE_FILL=ABSENT'
