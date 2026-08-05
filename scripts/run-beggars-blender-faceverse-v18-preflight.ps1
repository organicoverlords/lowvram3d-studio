[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$ProgressPreference = 'SilentlyContinue'

function Invoke-NativeChecked {
    param(
        [Parameter(Mandatory)][string]$FilePath,
        [Parameter(Mandatory)][string[]]$ArgumentList,
        [Parameter(Mandatory)][string]$FailureMessage
    )
    & $FilePath @ArgumentList
    $exitCode = $LASTEXITCODE
    if ($exitCode -ne 0) {
        throw "$FailureMessage (exit $exitCode)"
    }
}

$runId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { 'local' }
$output = "C:\AI\LowVRAM3D-benchmarks\beggars-scene\blender-faceverse-v18-preflight-$runId"
$inputRoot = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene\blender-faceverse-v11-preflight-30880575363'
$inputBlend = Join-Path $inputRoot 'beggars_photoreal_recreation.blend'
$reference = Join-Path $inputRoot 'projected_keyframe_031.png'
$baseline = Join-Path $inputRoot 'hero_clean_render.png'
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$upgrade = 'blender\upgrade_beggars_meme_scene_faceverse_v18.py'

foreach ($required in @($inputBlend, $reference, $baseline, $blender, $python, $upgrade)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "True-3D v18 input is missing: $required"
    }
}

$gpuLines = @(& nvidia-smi --query-compute-apps=used_memory --format=csv,noheader,nounits 2>$null)
$gpuUsed = 0
foreach ($line in $gpuLines) {
    $value = 0
    if ([int]::TryParse(([string]$line).Trim(), [ref]$value)) {
        $gpuUsed += $value
    }
}
Write-Host "GPU_USED_MIB=$gpuUsed"
if ($gpuUsed -ge 4800) {
    throw "GPU_BUSY: $gpuUsed MiB"
}

Invoke-NativeChecked -FilePath $python -ArgumentList @(
    '-m','py_compile',$upgrade
) -FailureMessage 'True-3D v18 Blender source failed compilation'

New-Item -ItemType Directory -Path $output -Force | Out-Null
Invoke-NativeChecked -FilePath $blender -ArgumentList @(
    '--background',
    '--factory-startup',
    '--python',$upgrade,
    '--',
    '--input-blend',$inputBlend,
    '--output-dir',$output,
    '--reference-image',$reference,
    '--baseline-image',$baseline
) -FailureMessage 'True-3D v18 Blender repair failed'

$requiredOutputs = @(
    'beggars_true3d_v18.blend',
    'public_reference.png',
    'v11_baseline.png',
    'v18_projection_042.png',
    'v18_projection_050.png',
    'v18_projection_058.png',
    'hero_v18.png',
    'wide_v18.png',
    'v18_report.json'
)
foreach ($name in $requiredOutputs) {
    $path = Join-Path $output $name
    if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
        throw "True-3D v18 output missing or empty: $name"
    }
}

$sheetScript = Join-Path $output 'make_v18_contact_sheet.py'
$sheetLines = @(
    'from pathlib import Path',
    'from PIL import Image, ImageDraw, ImageFont',
    'import os',
    'root = Path(os.environ["V18_OUTPUT"])',
    'items = [',
    '    ("public_reference.png", "PUBLIC REFERENCE"),',
    '    ("v11_baseline.png", "V11 TRUE-3D BASELINE"),',
    '    ("v18_projection_042.png", "V18 PROJECTION 0.42"),',
    '    ("v18_projection_050.png", "V18 PROJECTION 0.50"),',
    '    ("v18_projection_058.png", "V18 PROJECTION 0.58"),',
    '    ("hero_v18.png", "V18 SELECTED HERO"),',
    '    ("wide_v18.png", "V18 WIDE"),',
    ']',
    'tiles = []',
    'for name, label in items:',
    '    image = Image.open(root / name).convert("RGB")',
    '    image.thumbnail((640, 360), Image.Resampling.LANCZOS)',
    '    tile = Image.new("RGB", (640, 398), "black")',
    '    tile.paste(image, ((640-image.width)//2, 0))',
    '    ImageDraw.Draw(tile).text((12, 370), label, fill="white")',
    '    tiles.append(tile)',
    'rows = (len(tiles) + 1) // 2',
    'sheet = Image.new("RGB", (1280, rows * 398), "black")',
    'for index, tile in enumerate(tiles):',
    '    sheet.paste(tile, ((index % 2) * 640, (index // 2) * 398))',
    'sheet.save(root / "contact_sheet.png", quality=95)'
)
[System.IO.File]::WriteAllLines(
    $sheetScript,
    $sheetLines,
    [System.Text.UTF8Encoding]::new($false)
)
$env:V18_OUTPUT = $output
Invoke-NativeChecked -FilePath $python -ArgumentList @($sheetScript) -FailureMessage 'Could not build v18 contact sheet'
Remove-Item -LiteralPath $sheetScript -Force -ErrorAction SilentlyContinue

$contactSheet = Join-Path $output 'contact_sheet.png'
if (-not (Test-Path -LiteralPath $contactSheet) -or (Get-Item -LiteralPath $contactSheet).Length -lt 100000) {
    throw 'True-3D v18 contact sheet is missing or implausibly small'
}

if ($env:GITHUB_ENV) {
    "V18_OUTPUT=$output" | Add-Content -LiteralPath $env:GITHUB_ENV
}
Write-Host 'BLENDER_FACEVERSE_V18_TRUE3D_STILLS=PROVEN'
Write-Host "V18_OUTPUT=$output"
exit 0
