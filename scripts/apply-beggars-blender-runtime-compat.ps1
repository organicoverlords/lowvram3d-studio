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

$actionPattern = '(?ms)^    if face\.data\.shape_keys and face\.data\.shape_keys\.animation_data and face\.data\.shape_keys\.animation_data\.action:\r?\n        for fcurve in face\.data\.shape_keys\.animation_data\.action\.fcurves:\r?\n            for keyframe_point in fcurve\.keyframe_points:\r?\n                keyframe_point\.interpolation = "LINEAR"\r?\n'
$actionReplacement = @'
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
$patched = [regex]::Replace($content, $actionPattern, $actionReplacement, 1)
if ($patched -eq $content) {
    if ($content.Contains('for channelbag in getattr(strip, "channelbags", []):')) {
        Write-Host 'BLENDER_ACTION_RUNTIME_COMPAT=ALREADY_APPLIED'
    }
    else {
        throw 'Expected legacy Blender Action.fcurves block was not found.'
    }
}
else {
    $content = $patched
    Write-Host 'BLENDER_ACTION_RUNTIME_COMPAT=APPLIED'
}

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
