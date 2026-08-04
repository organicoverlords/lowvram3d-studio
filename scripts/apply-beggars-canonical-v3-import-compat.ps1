[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'blender\diagnose_beggars_canonical_face_v3.py'
$old = 'import diagnose_beggars_canonical_face_v2 as base'
$new = @'
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import diagnose_beggars_canonical_face_v2 as base
'@.TrimEnd()

if (-not (Test-Path -LiteralPath $target)) {
    throw "Canonical v3 source is missing: $target"
}
$text = Get-Content -LiteralPath $target -Raw
if ($text.Contains($old) -and -not $text.Contains('_SCRIPT_DIR = Path(__file__).resolve().parent')) {
    $text = $text.Replace($old, $new)
    [System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))
    Write-Host 'CANONICAL_V3_IMPORT_COMPAT=APPLIED'
}
elseif ($text.Contains('_SCRIPT_DIR = Path(__file__).resolve().parent')) {
    Write-Host 'CANONICAL_V3_IMPORT_COMPAT=ALREADY_APPLIED'
}
else {
    throw 'Canonical v3 helper import did not match the expected source form.'
}

$python = 'C:\AI\LowVRAM3D-cache\beggars-scene-v2\venv\Scripts\python.exe'
& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    throw 'Canonical v3 source failed compilation after import compatibility patch.'
}
Write-Host 'CANONICAL_V3_IMPORT_COMPILE=PROVEN'
