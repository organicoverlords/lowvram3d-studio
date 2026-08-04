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
    throw 'Checkout is visibly dirty before face-plate Action compatibility repair.'
}

$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))

$materialPattern = '(?ms)^    if material\.node_tree\.animation_data and material\.node_tree\.animation_data\.action:\r?\n        for fcurve in material\.node_tree\.animation_data\.action\.fcurves:\r?\n            for keyframe_point in fcurve\.keyframe_points:\r?\n                keyframe_point\.interpolation = "CONSTANT"\r?\n'
$materialReplacement = @'
    if material.node_tree.animation_data and material.node_tree.animation_data.action:
        action = material.node_tree.animation_data.action
        fcurves = list(getattr(action, "fcurves", []))
        if not fcurves:
            for layer in getattr(action, "layers", []):
                for strip in getattr(layer, "strips", []):
                    for channelbag in getattr(strip, "channelbags", []):
                        fcurves.extend(channelbag.fcurves)
        for fcurve in fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "CONSTANT"

'@
$patched = [regex]::Replace($content, $materialPattern, $materialReplacement, 1)
if ($patched -eq $content) {
    if ($content.Contains('action = material.node_tree.animation_data.action')) {
        Write-Host 'FACEPLATE_MATERIAL_ACTION_COMPAT=ALREADY_APPLIED'
    }
    else {
        throw 'Expected legacy face-sprite material Action.fcurves block was not found.'
    }
}
else {
    $content = $patched
    Write-Host 'FACEPLATE_MATERIAL_ACTION_COMPAT=APPLIED'
}

$platePattern = '(?ms)^    if plate\.data\.shape_keys and plate\.data\.shape_keys\.animation_data and plate\.data\.shape_keys\.animation_data\.action:\r?\n        for fcurve in plate\.data\.shape_keys\.animation_data\.action\.fcurves:\r?\n            for keyframe_point in fcurve\.keyframe_points:\r?\n                keyframe_point\.interpolation = "LINEAR"\r?\n'
$plateReplacement = @'
    if plate.data.shape_keys and plate.data.shape_keys.animation_data and plate.data.shape_keys.animation_data.action:
        action = plate.data.shape_keys.animation_data.action
        fcurves = list(getattr(action, "fcurves", []))
        if not fcurves:
            for layer in getattr(action, "layers", []):
                for strip in getattr(layer, "strips", []):
                    for channelbag in getattr(strip, "channelbags", []):
                        fcurves.extend(channelbag.fcurves)
        for fcurve in fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "LINEAR"

'@
$patched = [regex]::Replace($content, $platePattern, $plateReplacement, 1)
if ($patched -eq $content) {
    if ($content.Contains('action = plate.data.shape_keys.animation_data.action')) {
        Write-Host 'FACEPLATE_SHAPE_ACTION_COMPAT=ALREADY_APPLIED'
    }
    else {
        throw 'Expected legacy face-plate shape-key Action.fcurves block was not found.'
    }
}
else {
    $content = $patched
    Write-Host 'FACEPLATE_SHAPE_ACTION_COMPAT=APPLIED'
}

$legacyCount = @([regex]::Matches($content, '\.action\.fcurves')).Count
if ($legacyCount -ne 0) {
    throw "Legacy Blender Action.fcurves accesses remain after face-plate repair: $legacyCount"
}
if (-not $content.Contains('keyframe_point.interpolation = "CONSTANT"')) {
    throw 'Face-sprite cell animation is missing CONSTANT interpolation.'
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
    throw 'Face-plate Action-compatible Blender builder failed compile validation.'
}
& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not preserve the bounded face-plate Action compatibility repair.'
}
if (@(& git status --short).Count -gt 0) {
    throw 'Face-plate Action compatibility repair left visible checkout dirt.'
}

Write-Host 'FACEPLATE_BLENDER52_ACTION_COMPAT=PROVEN'
Write-Host 'FACEPLATE_LEGACY_ACTION_FCURVES_REMAINING=0'
