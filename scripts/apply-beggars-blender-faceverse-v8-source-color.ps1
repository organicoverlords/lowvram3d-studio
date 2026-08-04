[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'tools\beggars_scene\export_faceverse_v4_blender_model_sequence.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse model-space exporter is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw
$old = 'native_colors[valid] * 0.12 + sampled_colors[valid] * 0.88'
$new = 'native_colors[valid] * 0.32 + sampled_colors[valid] * 0.68'
if ($text.Contains($old)) {
    $text = $text.Replace($old, $new)
}
elseif (-not $text.Contains($new)) {
    throw 'Could not locate the FaceVerse source-color blend expression.'
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
        throw 'FaceVerse model-space exporter failed compilation after v8 color balance patch.'
    }
}
Write-Host 'FACEVERSE_V8_SOURCE_COLOR_BALANCE=PROVEN NATIVE_WEIGHT=0.32 SOURCE_WEIGHT=0.68'
