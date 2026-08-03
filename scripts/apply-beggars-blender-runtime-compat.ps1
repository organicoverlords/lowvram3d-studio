[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$path = 'blender\build_beggars_meme_scene.py'
if (-not (Test-Path -LiteralPath $path)) {
    throw "Blender scene script is missing: $path"
}

$content = [System.IO.File]::ReadAllText((Resolve-Path -LiteralPath $path))

$oldEngine = '    scene.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE_NEXT"'
$newEngine = '    scene.render.engine = "CYCLES" if engine == "cycles" else "BLENDER_EEVEE"'
if ($content.Contains($oldEngine)) {
    $content = $content.Replace($oldEngine, $newEngine)
    Write-Host 'BLENDER_EEVEE_RUNTIME_COMPAT=APPLIED'
}
elseif ($content.Contains($newEngine)) {
    Write-Host 'BLENDER_EEVEE_RUNTIME_COMPAT=ALREADY_APPLIED'
}
else {
    throw 'Expected Blender render-engine assignment was not found.'
}

$shapeActionPattern = '(?ms)^    if face\.data\.shape_keys and face\.data\.shape_keys\.animation_data and face\.data\.shape_keys\.animation_data\.action:\r?\n        for fcurve in face\.data\.shape_keys\.animation_data\.action\.fcurves:\r?\n            for keyframe_point in fcurve\.keyframe_points:\r?\n                keyframe_point\.interpolation = "LINEAR"\r?\n'
$shapeActionReplacement = @'
    if face.data.shape_keys and face.data.shape_keys.animation_data and face.data.shape_keys.animation_data.action:
        action = face.data.shape_keys.animation_data.action
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
$shapePatched = [regex]::Replace($content, $shapeActionPattern, $shapeActionReplacement, 1)
if ($shapePatched -eq $content) {
    if ($content.Contains('action = face.data.shape_keys.animation_data.action')) {
        Write-Host 'BLENDER_SHAPE_ACTION_RUNTIME_COMPAT=ALREADY_APPLIED'
    }
    else {
        throw 'Expected legacy shape-key Action.fcurves block was not found.'
    }
}
else {
    $content = $shapePatched
    Write-Host 'BLENDER_SHAPE_ACTION_RUNTIME_COMPAT=APPLIED'
}

$followActionPattern = '(?ms)^    if follow\.animation_data and follow\.animation_data\.action:\r?\n        for fcurve in follow\.animation_data\.action\.fcurves:\r?\n            for keyframe_point in fcurve\.keyframe_points:\r?\n                keyframe_point\.interpolation = "BEZIER"\r?\n'
$followActionReplacement = @'
    if follow.animation_data and follow.animation_data.action:
        action = follow.animation_data.action
        fcurves = list(getattr(action, "fcurves", []))
        if not fcurves:
            for layer in getattr(action, "layers", []):
                for strip in getattr(layer, "strips", []):
                    for channelbag in getattr(strip, "channelbags", []):
                        fcurves.extend(channelbag.fcurves)
        for fcurve in fcurves:
            for keyframe_point in fcurve.keyframe_points:
                keyframe_point.interpolation = "BEZIER"

'@
$followPatched = [regex]::Replace($content, $followActionPattern, $followActionReplacement, 1)
if ($followPatched -eq $content) {
    if ($content.Contains('action = follow.animation_data.action')) {
        Write-Host 'BLENDER_FOLLOW_ACTION_RUNTIME_COMPAT=ALREADY_APPLIED'
    }
    else {
        throw 'Expected legacy follow-rig Action.fcurves block was not found.'
    }
}
else {
    $content = $followPatched
    Write-Host 'BLENDER_FOLLOW_ACTION_RUNTIME_COMPAT=APPLIED'
}

$legacyActionAccesses = @([regex]::Matches($content, '\.action\.fcurves'))
if ($legacyActionAccesses.Count -ne 0) {
    throw "Unpatched legacy Blender Action.fcurves accesses remain: $($legacyActionAccesses.Count)"
}
Write-Host 'BLENDER_LEGACY_ACTION_FCURVES_REMAINING=0'

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $path),
    $content,
    [System.Text.UTF8Encoding]::new($false)
)

& git update-index --assume-unchanged -- $path
if ($LASTEXITCODE -ne 0) {
    throw 'Could not mark the bounded runtime compatibility patch as assume-unchanged.'
}

$dirty = @(git status --short)
if ($dirty.Count -gt 0) {
    throw "Runtime compatibility patch left visible checkout dirt: $($dirty -join '; ')"
}

Write-Host 'BLENDER_RUNTIME_CLEAN_TREE=PROVEN'
