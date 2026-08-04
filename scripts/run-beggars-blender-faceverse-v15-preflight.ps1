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
    if ($LASTEXITCODE -ne 0) { throw "$FailureMessage (exit $LASTEXITCODE)" }
}

$runId = if ($env:GITHUB_RUN_ID) { $env:GITHUB_RUN_ID } else { 'local' }
$root = 'C:\AI\LowVRAM3D-benchmarks\beggars-scene'
$v11Output = Join-Path $root "blender-faceverse-v11-preflight-$runId"
$output = Join-Path $root "blender-faceverse-v15-preflight-$runId"
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$v11Runner = 'scripts\run-beggars-blender-faceverse-v11-preflight.ps1'
$cutoutBuilder = 'tools\beggars_scene\build_source_head_cutout.py'
$overlayBuilder = 'tools\beggars_scene\build_source_head_overlays.py'
$v15Builder = 'blender\upgrade_beggars_meme_scene_faceverse_v15.py'

foreach ($required in @($blender,$python,$v11Runner,$cutoutBuilder,$overlayBuilder,$v15Builder)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Blender v15 input missing: $required" }
}

# Blender 5.x changed Alpha Over's socket layout. Never use positional input 0.
$v15Text = Get-Content -LiteralPath $v15Builder -Raw
$unsafeFactor = '    alpha_over.inputs[0].default_value = 1.0'
$safeFactor = @'
    factor_socket = alpha_over.inputs.get("Factor") or alpha_over.inputs.get("Fac")
    if factor_socket is not None and hasattr(factor_socket, "default_value"):
        try:
            factor_socket.default_value = 1.0
        except TypeError:
            pass
'@
if ($v15Text.Contains($unsafeFactor)) {
    $v15Text = $v15Text.Replace($unsafeFactor, $safeFactor.TrimEnd("`r","`n"))
}
elseif (-not $v15Text.Contains('factor_socket = alpha_over.inputs.get("Factor")')) {
    throw 'Could not locate the Blender 5 Alpha Over compatibility anchor.'
}
[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $v15Builder),
    $v15Text,
    [System.Text.UTF8Encoding]::new($false)
)
Write-Host 'BLENDER_FACEVERSE_V15_ALPHA_OVER_SOCKET_COMPAT=PROVEN'

Invoke-NativeChecked $python @('-m','py_compile',$cutoutBuilder,$overlayBuilder,$v15Builder) 'Blender v15 sources failed compilation'
Invoke-NativeChecked 'powershell.exe' @('-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$v11Runner) 'Proven Blender v11 prerequisite failed'

$v11Blend = Join-Path $v11Output 'beggars_photoreal_recreation.blend'
$v11SourceImage = Join-Path $v11Output 'projected_keyframe_031.png'
$v11Receipt = Join-Path $v11Output 'scene_receipt.json'
$v11Rejected = Join-Path $v11Output 'projection_blend_060.png'
foreach ($required in @($v11Blend,$v11SourceImage,$v11Receipt,$v11Rejected)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "V15 prerequisite missing: $required" }
}

if (Test-Path $output) { Remove-Item $output -Recurse -Force }
New-Item -ItemType Directory -Path $output -Force | Out-Null
$overlayDir = Join-Path $output 'overlays'
$cutout = Join-Path $output 'source_head_cutout.png'
$cutoutReport = Join-Path $output 'source_head_cutout_report.json'
$overlayReport = Join-Path $output 'source_head_overlay_report.json'

Invoke-NativeChecked $python @($cutoutBuilder,'--input',$v11SourceImage,'--output',$cutout,'--report',$cutoutReport,'--crop','155,18,405,250','--feather','4.0') 'Source head cutout failed'
Invoke-NativeChecked $python @($overlayBuilder,'--cutout',$cutout,'--output-dir',$overlayDir,'--report',$overlayReport,'--width','1280','--height','720') 'Full-frame overlay generation failed'
Copy-Item $v11SourceImage (Join-Path $output 'projected_keyframe_031.png') -Force

Invoke-NativeChecked $blender @('--background',$v11Blend,'--python',$v15Builder,'--','--output-dir',$output,'--overlay-dir',$overlayDir,'--overlay-report',$overlayReport,'--cutout-report',$cutoutReport,'--base-receipt',$v11Receipt,'--frame','48') 'Blender compositor v15 failed'

$requiredNames = @(
 'beggars_photoreal_recreation.blend','hero_composited_head_render.png','scene_receipt.json','v15_visual_report.json',
 'source_head_cutout.png','source_head_cutout_report.json','source_head_overlay_report.json','projected_keyframe_031.png',
 'composite_head_scale_185.png','composite_head_scale_200.png','composite_head_scale_215.png',
 'composite_head_left_350.png','composite_head_right_410.png','composite_head_high_030.png','composite_head_low_070.png'
)
foreach ($name in $requiredNames) {
    $path = Join-Path $output $name
    if (-not (Test-Path $path) -or (Get-Item $path).Length -le 0) { throw "V15 output missing: $name" }
}

$sheetScript = Join-Path $output 'make_contact_sheet.py'
@'
from pathlib import Path
from PIL import Image, ImageDraw
import os
root=Path(os.environ['B15_OUTPUT']); v11=Path(os.environ['B15_V11_OUTPUT'])
items=[
(root/'projected_keyframe_031.png','PUBLIC REFERENCE'),(v11/'projection_blend_060.png','V11 REJECTED'),
(root/'composite_head_scale_185.png','V15 SCALE 185'),(root/'composite_head_scale_200.png','V15 SCALE 200'),
(root/'composite_head_scale_215.png','V15 SCALE 215'),(root/'composite_head_left_350.png','V15 LEFT'),
(root/'composite_head_right_410.png','V15 RIGHT'),(root/'composite_head_high_030.png','V15 HIGH'),
(root/'composite_head_low_070.png','V15 LOW'),(root/'hero_composited_head_render.png','V15 SELECTED RELOAD')]
tiles=[]
for p,label in items:
 im=Image.open(p).convert('RGB'); im.thumbnail((640,360),Image.Resampling.LANCZOS)
 tile=Image.new('RGB',(640,404),'black'); tile.paste(im,((640-im.width)//2,0)); ImageDraw.Draw(tile).text((12,374),label,fill='white'); tiles.append(tile)
sheet=Image.new('RGB',(1280,2020),'black')
for i,t in enumerate(tiles): sheet.paste(t,((i%2)*640,(i//2)*404))
sheet.save(root/'contact_sheet.png',quality=95)
'@ | Set-Content -LiteralPath $sheetScript -Encoding utf8
$env:B15_OUTPUT=$output; $env:B15_V11_OUTPUT=$v11Output
Invoke-NativeChecked $python @($sheetScript) 'V15 contact sheet failed'
Remove-Item $sheetScript -Force
if ((Get-Item (Join-Path $output 'contact_sheet.png')).Length -lt 100000) { throw 'V15 contact sheet too small' }
if ($env:GITHUB_ENV) { "B15_OUTPUT=$output" | Add-Content $env:GITHUB_ENV }
Write-Host "BLENDER_FACEVERSE_V15_OUTPUT=$output"
Write-Host 'BLENDER_FACEVERSE_V15_RUNNER=PROVEN_USER_VISUAL_REVIEW_REQUIRED'
