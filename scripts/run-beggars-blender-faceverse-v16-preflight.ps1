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
$output = Join-Path $root "blender-faceverse-v16-preflight-$runId"
$blender = 'C:\Program Files\Blender Foundation\Blender 5.2\blender.exe'
$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$v11Runner = 'scripts\run-beggars-blender-faceverse-v11-preflight.ps1'
$cutoutBuilder = 'tools\beggars_scene\build_source_head_cutout.py'
$overlayBuilder = 'tools\beggars_scene\build_source_head_overlays.py'
$compositor = 'tools\beggars_scene\composite_head_overlays_v16.py'
$bodyBuilder = 'blender\prepare_beggars_body_plate_v16.py'

foreach ($required in @($blender,$python,$v11Runner,$cutoutBuilder,$overlayBuilder,$compositor,$bodyBuilder)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Blender v16 input missing: $required" }
}
Invoke-NativeChecked $python @('-m','py_compile',$cutoutBuilder,$overlayBuilder,$compositor,$bodyBuilder) 'Blender v16 sources failed compilation'
Invoke-NativeChecked 'powershell.exe' @('-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$v11Runner) 'Proven Blender v11 prerequisite failed'

$v11Blend = Join-Path $v11Output 'beggars_photoreal_recreation.blend'
$v11SourceImage = Join-Path $v11Output 'projected_keyframe_031.png'
$v11Receipt = Join-Path $v11Output 'scene_receipt.json'
foreach ($required in @($v11Blend,$v11SourceImage,$v11Receipt)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "V16 prerequisite missing: $required" }
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

Invoke-NativeChecked $blender @('--background',$v11Blend,'--python',$bodyBuilder,'--','--output-dir',$output,'--base-receipt',$v11Receipt,'--frame','48') 'Blender v16 body plate failed'
Invoke-NativeChecked $python @($compositor,'--body-plate',(Join-Path $output 'body_plate.png'),'--overlay-dir',$overlayDir,'--overlay-report',$overlayReport,'--output-dir',$output,'--selected','scale_200') 'External v16 compositing failed'

$requiredNames = @(
 'beggars_photoreal_recreation.blend','body_plate.png','hero_external_composite.png','contact_sheet.png',
 'scene_receipt.json','v16_visual_report.json','source_head_cutout.png','source_head_cutout_report.json',
 'source_head_overlay_report.json','projected_keyframe_031.png',
 'external_composite_scale_185.png','external_composite_scale_200.png','external_composite_scale_215.png',
 'external_composite_left_350.png','external_composite_right_410.png','external_composite_high_030.png','external_composite_low_070.png'
)
foreach ($name in $requiredNames) {
    $path = Join-Path $output $name
    if (-not (Test-Path $path) -or (Get-Item $path).Length -le 0) { throw "V16 output missing: $name" }
}
if ((Get-Item (Join-Path $output 'contact_sheet.png')).Length -lt 100000) { throw 'V16 contact sheet too small' }
if ($env:GITHUB_ENV) { "B16_OUTPUT=$output" | Add-Content $env:GITHUB_ENV }
Write-Host "BLENDER_FACEVERSE_V16_OUTPUT=$output"
Write-Host 'BLENDER_FACEVERSE_V16_RUNNER=PROVEN_USER_VISUAL_REVIEW_REQUIRED'
