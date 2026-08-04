[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\build_beggars_meme_scene_faceverse_v7.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse v7 Blender source is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$moduleAnchor = @'
base = v6.base
v5 = v6.v5
'@
$moduleReplacement = @'
base = v6.base
v5 = v6.v5
_original_create_model_space_face_mesh = v5.create_model_space_face_mesh
_original_build_refined_character = v6.build_refined_character
'@
if ($text.Contains($moduleAnchor)) {
    $text = $text.Replace($moduleAnchor, $moduleReplacement)
}
elseif (-not $text.Contains('_original_create_model_space_face_mesh')) {
    throw 'Could not locate the v7 original-function capture anchor.'
}

$faceCall = '    face, follow, targets = v5.create_model_space_face_mesh(*args, **kwargs)'
$faceReplacement = '    face, follow, targets = _original_create_model_space_face_mesh(*args, **kwargs)'
if ($text.Contains($faceCall)) {
    $text = $text.Replace($faceCall, $faceReplacement)
}
elseif (-not $text.Contains($faceReplacement)) {
    throw 'Could not locate the v7 face-wrapper recursion anchor.'
}

$characterCall = '    result = v6.build_refined_character(follow, colors_rgb)'
$characterReplacement = '    result = _original_build_refined_character(follow, colors_rgb)'
if ($text.Contains($characterCall)) {
    $text = $text.Replace($characterCall, $characterReplacement)
}
elseif (-not $text.Contains($characterReplacement)) {
    throw 'Could not locate the v7 character-wrapper recursion anchor.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $target),
    $text,
    [System.Text.UTF8Encoding]::new($false)
)

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "FaceVerse Python is missing: $python"
}
& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    throw 'FaceVerse v7 Blender source failed compilation after recursion repair.'
}

$patched = Get-Content -LiteralPath $target -Raw
if ($patched.Contains('face, follow, targets = v5.create_model_space_face_mesh')) {
    throw 'FaceVerse v7 face recursion remains after patch.'
}
if ($patched.Contains('result = v6.build_refined_character(follow, colors_rgb)')) {
    throw 'FaceVerse v7 character recursion remains after patch.'
}
if (-not $patched.Contains('_original_create_model_space_face_mesh')) {
    throw 'FaceVerse v7 original face function was not captured.'
}
if (-not $patched.Contains('_original_build_refined_character')) {
    throw 'FaceVerse v7 original character function was not captured.'
}

Write-Host 'BLENDER_FACEVERSE_V7_RECURSION_COMPAT=PROVEN'
