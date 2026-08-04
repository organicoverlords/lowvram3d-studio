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
$output = Join-Path $root "blender-faceverse-v14-preflight-$runId"
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$v11Runner = 'scripts\run-beggars-blender-faceverse-v11-preflight.ps1'
$cutoutBuilder = 'tools\beggars_scene\build_source_head_cutout.py'
$v14Builder = 'blender\upgrade_beggars_meme_scene_faceverse_v14.py'

foreach ($required in @($blender, $python, $v11Runner, $cutoutBuilder, $v14Builder)) {
    if (-not (Test-Path -LiteralPath $required)) {
        throw "Blender v14 input missing: $required"
    }
}

Invoke-NativeChecked -FilePath $python -ArgumentList @(
    '-m','py_compile',$cutoutBuilder,$v14Builder
) -FailureMessage 'Blender v14 sources failed compilation'

Invoke-NativeChecked -FilePath 'powershell.exe' -ArgumentList @(
    '-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$v11Runner
) -FailureMessage 'Proven Blender v11 prerequisite run failed'

$v11Blend = Join-Path $v11Output 'beggars_photoreal_recreation.blend'
$v11SourceImage = Join-Path $v11Output 'projected_keyframe_031.png'
$v11Receipt = Join-Path $v11Output 'scene_receipt.json'
$v11Rejected = Join-Path $v11Output 'projection_blend_060.png'
foreach ($required in @($v11Blend, $v11SourceImage, $v11Receipt, $v11Rejected)) {
    if (-not (Test-Path -LiteralPath $required) -or (Get-Item -LiteralPath $required).Length -le 0) {
        throw "Blender v14 prerequisite output missing or empty: $required"
    }
}

if (Test-Path -LiteralPath $output) {
    Remove-Item -LiteralPath $output -Recurse -Force
}
New-Item -ItemType Directory -Path $output -Force | Out-Null

$cutout = Join-Path $output 'source_head_cutout.png'
$cutoutReport = Join-Path $output 'source_head_cutout_report.json'
Invoke-NativeChecked -FilePath $python -ArgumentList @(
    $cutoutBuilder,
    '--input',$v11SourceImage,
    '--output',$cutout,
    '--report',$cutoutReport,
    '--crop','155,18,405,250',
    '--feather','4.0'
) -FailureMessage 'Could not rebuild the feathered source head cutout'

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
    '--python',$v14Builder,
    '--',
    '--output-dir',$output,
    '--cutout',$cutout,
    '--cutout-report',$cutoutReport,
    '--base-receipt',$v11Receipt,
    '--frame','48'
) -FailureMessage 'Opaque camera-space Blender v14 preflight failed'

foreach ($name in @(
    'beggars_photoreal_recreation.blend',
    'hero_camera_head_render.png',
    'camera_head_scale_090.png',
    'camera_head_scale_096.png',
    'camera_head_scale_102.png',
    'camera_head_left_035.png',
    'camera_head_right_005.png',
    'camera_head_low_010.png',
    'camera_head_high_040.png',
    'source_head_cutout.png',
    'source_head_cutout_report.json',
    'scene_receipt.json',
    'v14_visual_report.json',
    'projected_keyframe_031.png'
)) {
    $path = Join-Path $output $name
    if (-not (Test-Path -LiteralPath $path) -or (Get-Item -LiteralPath $path).Length -le 0) {
        throw "Opaque camera-space Blender v14 output missing or empty: $name"
    }
}
Write-Host 'BLENDER_FACEVERSE_V14_SAVE_RELOAD_STILLS=PROVEN'

$sheetScript = Join-Path $output 'make_contact_sheet.py'
$sheetLines = @(
    'from pathlib import Path',
    'from PIL import Image, ImageDraw',
    'import os',
    'output = Path(os.environ["B14_OUTPUT"])',
    'v11 = Path(os.environ["B14_V11_OUTPUT"])',
    'items = [',
    '    (output / "projected_keyframe_031.png", "PUBLIC REFERENCE KEYFRAME"),',
    '    (v11 / "projection_blend_060.png", "V11 REJECTED SYNTHETIC HEAD"),',
    '    (output / "camera_head_scale_090.png", "V14 SCALE 90%"),',
    '    (output / "camera_head_scale_096.png", "V14 SCALE 96%"),',
    '    (output / "camera_head_scale_102.png", "V14 SCALE 102%"),',
    '    (output / "camera_head_left_035.png", "V14 X -0.35"),',
    '    (output / "camera_head_right_005.png", "V14 X -0.05"),',
    '    (output / "camera_head_low_010.png", "V14 Y 0.10"),',
    '    (output / "camera_head_high_040.png", "V14 Y 0.40"),',
    '    (output / "hero_camera_head_render.png", "V14 SELECTED SAVE/RELOAD"),',
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
    'sheet = Image.new("RGB", (1280, 2020), "black")',
    'for index, tile in enumerate(tiles):',
    '    sheet.paste(tile, ((index % 2) * 640, (index // 2) * 404))',
    'sheet.save(output / "contact_sheet.png", quality=95)'
)
[System.IO.File]::WriteAllLines(
    $sheetScript,
    $sheetLines,
    [System.Text.UTF8Encoding]::new($false)
)
$env:B14_OUTPUT = $output
$env:B14_V11_OUTPUT = $v11Output
Invoke-NativeChecked -FilePath $python -ArgumentList @($sheetScript) -FailureMessage 'Could not build Blender v14 contact sheet'
Remove-Item -LiteralPath $sheetScript -Force -ErrorAction SilentlyContinue

$contactSheet = Join-Path $output 'contact_sheet.png'
if (-not (Test-Path -LiteralPath $contactSheet) -or (Get-Item -LiteralPath $contactSheet).Length -lt 100000) {
    throw 'Blender v14 contact sheet is missing or implausibly small'
}

if ($env:GITHUB_ENV) {
    "B14_OUTPUT=$output" | Add-Content -LiteralPath $env:GITHUB_ENV
}
Write-Host "BLENDER_FACEVERSE_V14_OUTPUT=$output"
Write-Host 'BLENDER_FACEVERSE_V14_RUNNER=PROVEN_USER_VISUAL_REVIEW_REQUIRED'
exit 0
