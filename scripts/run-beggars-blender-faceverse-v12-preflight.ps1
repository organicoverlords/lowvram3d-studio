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
$root = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene'
$v11Output = Join-Path $root "blender-faceverse-v11-preflight-$runId"
$output = Join-Path $root "blender-faceverse-v12-preflight-$runId"
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$v11Runner = 'scripts\run-beggars-blender-faceverse-v11-preflight.ps1'
$v12Builder = 'blender\upgrade_beggars_meme_scene_faceverse_v12.py'

foreach ($required in @($blender, $python, $v11Runner, $v12Builder)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Blender v12 input missing: $required"
    }
}

Invoke-NativeChecked -FilePath $python -ArgumentList @(
    '-m','py_compile',$v12Builder
) -FailureMessage 'Blender v12 source failed compilation'

Invoke-NativeChecked -FilePath 'powershell.exe' -ArgumentList @(
    '-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$v11Runner
) -FailureMessage 'Proven Blender v11 prerequisite run failed'

$v11Blend = Join-Path $v11Output 'beggars_photoreal_recreation.blend'
$v11SourceImage = Join-Path $v11Output 'projected_keyframe_031.png'
$v11Receipt = Join-Path $v11Output 'scene_receipt.json'
foreach ($required in @($v11Blend, $v11SourceImage, $v11Receipt)) {
    if (-not (Test-Path -LiteralPath $required) -or (Get-Item -LiteralPath $required).Length -le 0) {
        throw "Blender v12 prerequisite output missing or empty: $required"
    }
}

if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Recurse -Force
}
New-Item -ItemType Directory -Path $output -Force | Out-Null

foreach ($name in @(
    'projected_keyframe_031.png',
    'faceverse_model_space_sequence_report.json',
    'projection_texture_report.json'
)) {
    $source = Join-Path $v11Output $name
    if (Test-Path -LiteralPath $source) {
        Copy-Item -LiteralPath $source -Destination (Join-Path $output $name) -Force
    }
}

Invoke-NativeChecked -FilePath $blender -ArgumentList @(
    '--background',$v11Blend,
    '--python',$v12Builder,
    '--',
    '--output-dir',$output,
    '--source-image',$v11SourceImage,
    '--base-receipt',$v11Receipt,
    '--crop-x0','138',
    '--crop-y0','18',
    '--crop-x1','407',
    '--crop-y1','249',
    '--frame','48'
) -FailureMessage 'Source-aligned Blender v12 preflight failed'

foreach ($name in @(
    'beggars_photoreal_recreation.blend',
    'hero_source_shell_render.png',
    'source_shell_094.png',
    'source_shell_100.png',
    'source_shell_106.png',
    'scene_receipt.json',
    'v12_visual_report.json',
    'projected_keyframe_031.png'
)) {
    $path = Join-Path $output $name
    if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
        throw "Source-aligned Blender v12 output missing or empty: $name"
    }
}
Write-Host 'BLENDER_FACEVERSE_V12_SAVE_RELOAD_STILLS=PROVEN'

$sheetScript = Join-Path $output 'make_contact_sheet.py'
$sheetLines = @(
    'from pathlib import Path',
    'from PIL import Image, ImageDraw',
    'import os',
    'output = Path(os.environ["B12_OUTPUT"])',
    'v11 = Path(os.environ["B12_V11_OUTPUT"])',
    'items = [',
    '    (output / "projected_keyframe_031.png", "PUBLIC REFERENCE KEYFRAME"),',
    '    (v11 / "projection_blend_060.png", "V11 REJECTED: UV PROJECTION"),',
    '    (output / "source_shell_094.png", "V12 SOURCE SHELL 94%"),',
    '    (output / "source_shell_100.png", "V12 SOURCE SHELL 100%"),',
    '    (output / "source_shell_106.png", "V12 SOURCE SHELL 106%"),',
    '    (output / "hero_source_shell_render.png", "V12 SELECTED SAVE/RELOAD"),',
    ']',
    'tiles = []',
    'for path, label in items:',
    '    image = Image.open(path).convert("RGB")',
    '    image.thumbnail((640, 360), Image.Resampling.LANCZOS)',
    '    tile = Image.new("RGB", (640, 404), "black")',
    '    tile.paste(image, ((640 - image.width) // 2, 0))',
    '    draw = ImageDraw.Draw(tile)',
    '    draw.rectangle((0, 360, 640, 404), fill="black")',
    '    draw.text((12, 374), label, fill="white")',
    '    tiles.append(tile)',
    'sheet = Image.new("RGB", (1280, 1212), "black")',
    'for index, tile in enumerate(tiles):',
    '    sheet.paste(tile, ((index % 2) * 640, (index // 2) * 404))',
    'sheet.save(output / "contact_sheet.png", quality=95)'
)
[System.IO.File]::WriteAllLines(
    $sheetScript,
    $sheetLines,
    [System.Text.UTF8Encoding]::new($false)
)
$env:B12_OUTPUT = $output
$env:B12_V11_OUTPUT = $v11Output
Invoke-NativeChecked -FilePath $python -ArgumentList @($sheetScript) -FailureMessage 'Could not build Blender v12 contact sheet'
Remove-Item -LiteralPath $sheetScript -Force -ErrorAction SilentlyContinue

$contactSheet = Join-Path $output 'contact_sheet.png'
if (-not (Test-Path -LiteralPath $contactSheet) -or (Get-Item -LiteralPath $contactSheet).Length -lt 100000) {
    throw 'Blender v12 contact sheet is missing or implausibly small'
}

if ($env:GITHUB_ENV) {
    "B12_OUTPUT=$output" | Add-Content -LiteralPath $env:GITHUB_ENV
}
Write-Host "BLENDER_FACEVERSE_V12_OUTPUT=$output"
Write-Host 'BLENDER_FACEVERSE_V12_RUNNER=PROVEN_USER_VISUAL_REVIEW_REQUIRED'
exit 0
