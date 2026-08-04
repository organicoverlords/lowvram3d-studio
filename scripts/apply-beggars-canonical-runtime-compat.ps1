[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\diagnose_beggars_canonical_face.py'
$old = 'scene.world.color = (0.003, 0.0015, 0.001, 1.0)'
$new = 'scene.world.color = (0.003, 0.0015, 0.001)'

if (-not (Test-Path -LiteralPath $target)) {
    throw "Canonical Blender diagnostic source is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw
if ($text.Contains($old)) {
    $text = $text.Replace($old, $new)
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))
    Write-Host 'CANONICAL_WORLD_COLOR_COMPAT=APPLIED'
}
elseif ($text.Contains($new)) {
    Write-Host 'CANONICAL_WORLD_COLOR_COMPAT=ALREADY_APPLIED'
}
else {
    throw 'Canonical world-color assignment did not match the expected source form.'
}

$python = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2\venv\Scripts\python.exe'
& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    throw 'Canonical Blender source failed compilation after compatibility patch.'
}

Write-Host 'CANONICAL_RUNTIME_PATCH_COMPILE=PROVEN'
