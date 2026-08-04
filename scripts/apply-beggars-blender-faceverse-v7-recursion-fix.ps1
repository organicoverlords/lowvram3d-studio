[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\build_beggars_meme_scene_faceverse_v7.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "Blender v7 scene source is missing: $target"
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
elseif (
    -not $text.Contains('_original_create_model_space_face_mesh') -or
    -not $text.Contains('_original_build_refined_character')
) {
    throw 'Could not locate the Blender v7 original-function binding anchor.'
}

$oldFaceCall = '    face, follow, targets = v5.create_model_space_face_mesh(*args, **kwargs)'
$newFaceCall = '    face, follow, targets = _original_create_model_space_face_mesh(*args, **kwargs)'
if ($text.Contains($oldFaceCall)) {
    $text = $text.Replace($oldFaceCall, $newFaceCall)
}
elseif (-not $text.Contains($newFaceCall)) {
    throw 'Could not locate the Blender v7 recursive face-constructor call.'
}

$oldCharacterCall = '    result = v6.build_refined_character(follow, colors_rgb)'
$newCharacterCall = '    result = _original_build_refined_character(follow, colors_rgb)'
if ($text.Contains($oldCharacterCall)) {
    $text = $text.Replace($oldCharacterCall, $newCharacterCall)
}
elseif (-not $text.Contains($newCharacterCall)) {
    throw 'Could not locate the Blender v7 recursive character-builder call.'
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
    throw 'Blender v7 scene failed compilation after recursion repair.'
}

$patched = Get-Content -LiteralPath $target -Raw
if ($patched.Contains('face, follow, targets = v5.create_model_space_face_mesh')) {
    throw 'Blender v7 recursive face-constructor call remains.'
}
if ($patched.Contains('result = v6.build_refined_character(follow, colors_rgb)')) {
    throw 'Blender v7 recursive character-builder call remains.'
}
if (-not $patched.Contains('_original_create_model_space_face_mesh')) {
    throw 'Blender v7 original face constructor is not bound.'
}
if (-not $patched.Contains('_original_build_refined_character')) {
    throw 'Blender v7 original character builder is not bound.'
}

Write-Host 'BLENDER_FACEVERSE_V7_ORIGINAL_FUNCTION_BINDINGS=PROVEN'
