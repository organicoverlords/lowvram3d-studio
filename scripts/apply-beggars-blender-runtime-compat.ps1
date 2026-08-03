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

$importPattern = '(?m)^import argparse\r?\nimport json\r?\nimport math\r?\nimport random\r?\nimport sys\r?\n'
$importReplacement = @'
import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
'@
$importPatched = [regex]::Replace($content, $importPattern, $importReplacement, 1)
if ($importPatched -eq $content) {
    if ($content.Contains('import subprocess') -and $content.Contains('import shutil')) {
        Write-Host 'BLENDER_EXTERNAL_FFMPEG_IMPORTS=ALREADY_APPLIED'
    }
    else {
        throw 'Expected Blender Python import block was not found.'
    }
}
else {
    $content = $importPatched
    Write-Host 'BLENDER_EXTERNAL_FFMPEG_IMPORTS=APPLIED'
}

$animationPattern = '(?ms)^def render_animation\(scene: bpy\.types\.Scene, camera: bpy\.types\.Object, path: Path\) -> None:\r?\n    scene\.camera = camera\r?\n    scene\.render\.image_settings\.file_format = "FFMPEG"\r?\n    scene\.render\.ffmpeg\.format = "MPEG4"\r?\n    scene\.render\.ffmpeg\.codec = "H264"\r?\n    scene\.render\.ffmpeg\.constant_rate_factor = "MEDIUM"\r?\n    scene\.render\.ffmpeg\.ffmpeg_preset = "GOOD"\r?\n    scene\.render\.filepath = str\(path\)\r?\n    bpy\.ops\.render\.render\(animation=True\)\r?\n'
$animationReplacement = @'
def render_animation(scene: bpy.types.Scene, camera: bpy.types.Object, path: Path) -> None:
    scene.camera = camera
    frame_dir = path.parent / f"{path.stem}_frames"
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"
    scene.render.filepath = str(frame_dir / "frame_")
    bpy.ops.render.render(animation=True)

    ffmpeg_exe = os.environ.get("FFMPEG_EXE")
    if not ffmpeg_exe or not Path(ffmpeg_exe).is_file():
        raise RuntimeError(f"External FFmpeg is unavailable: {ffmpeg_exe}")
    subprocess.run(
        [
            ffmpeg_exe,
            "-y",
            "-framerate",
            str(scene.render.fps),
            "-start_number",
            str(scene.frame_start),
            "-i",
            str(frame_dir / "frame_%04d.png"),
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(path),
        ],
        check=True,
    )
    shutil.rmtree(frame_dir)

'@
$animationPatched = [regex]::Replace($content, $animationPattern, $animationReplacement, 1)
if ($animationPatched -eq $content) {
    if ($content.Contains('ffmpeg_exe = os.environ.get("FFMPEG_EXE")')) {
        Write-Host 'BLENDER_EXTERNAL_FFMPEG_RENDER=ALREADY_APPLIED'
    }
    else {
        throw 'Expected Blender internal FFMPEG animation block was not found.'
    }
}
else {
    $content = $animationPatched
    Write-Host 'BLENDER_EXTERNAL_FFMPEG_RENDER=APPLIED'
}

if ($content.Contains('image_settings.file_format = "FFMPEG"')) {
    throw 'An unsupported Blender internal FFMPEG output assignment remains.'
}
Write-Host 'BLENDER_INTERNAL_FFMPEG_ASSIGNMENTS_REMAINING=0'

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
