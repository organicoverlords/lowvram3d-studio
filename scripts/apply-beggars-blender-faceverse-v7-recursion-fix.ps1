[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\build_beggars_meme_scene_faceverse_v7.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "Blender v7 scene source is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw
$binding = '_original_create_model_space_face_mesh = v5.create_model_space_face_mesh'
if (-not $text.Contains($binding)) {
    $anchor = "base = v6.base`nv5 = v6.v5`n"
    if (-not $text.Contains($anchor)) {
        throw 'Could not locate the Blender v7 original-constructor binding anchor.'
    }
    $text = $text.Replace(
        $anchor,
        $anchor + $binding + "`n"
    )
}

$oldCall = '    face, follow, targets = v5.create_model_space_face_mesh(*args, **kwargs)'
$newCall = '    face, follow, targets = _original_create_model_space_face_mesh(*args, **kwargs)'
if ($text.Contains($oldCall)) {
    $text = $text.Replace($oldCall, $newCall)
}
elseif (-not $text.Contains($newCall)) {
    throw 'Could not locate the Blender v7 recursive constructor call.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $target),
    $text,
    [System.Text.UTF8Encoding]::new($false)
)

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
if (Test-Path -LiteralPath $python) {
    & $python -m py_compile $target
    if ($LASTEXITCODE -ne 0) {
        throw 'Blender v7 scene failed compilation after recursion fix.'
    }
}
Write-Host 'BLENDER_FACEVERSE_V7_ORIGINAL_CONSTRUCTOR_BINDING=PROVEN'
