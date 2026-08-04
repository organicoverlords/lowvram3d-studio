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
$v16Output = Join-Path $root "blender-faceverse-v16-preflight-$runId"
$output = Join-Path $root "blender-faceverse-v17-preflight-$runId"
$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
$v16Runner = 'scripts\run-beggars-blender-faceverse-v16-preflight.ps1'
$v17Compositor = 'tools\beggars_scene\composite_head_bridge_v17.py'

foreach ($required in @($python,$v16Runner,$v17Compositor)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Blender v17 input missing: $required" }
}
Invoke-NativeChecked $python @('-m','py_compile',$v17Compositor) 'Blender v17 source failed compilation'
Invoke-NativeChecked 'powershell.exe' @('-NoLogo','-NoProfile','-ExecutionPolicy','Bypass','-File',$v16Runner) 'Proven Blender v16 prerequisite failed'

$body = Join-Path $v16Output 'body_plate.png'
$head = Join-Path $v16Output 'source_head_cutout.png'
$reference = Join-Path $v16Output 'projected_keyframe_031.png'
$blend = Join-Path $v16Output 'beggars_photoreal_recreation.blend'
$receipt = Join-Path $v16Output 'scene_receipt.json'
foreach ($required in @($body,$head,$reference,$blend,$receipt)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "V17 prerequisite missing: $required" }
}

if (Test-Path $output) { Remove-Item $output -Recurse -Force }
New-Item -ItemType Directory -Path $output -Force | Out-Null
Copy-Item $body (Join-Path $output 'body_plate.png') -Force
Copy-Item $head (Join-Path $output 'source_head_cutout.png') -Force
Copy-Item $reference (Join-Path $output 'projected_keyframe_031.png') -Force
Copy-Item $blend (Join-Path $output 'beggars_photoreal_recreation.blend') -Force
Copy-Item $receipt (Join-Path $output 'scene_receipt.json') -Force

Invoke-NativeChecked $python @(
    $v17Compositor,
    '--body-plate',(Join-Path $output 'body_plate.png'),
    '--head-cutout',(Join-Path $output 'source_head_cutout.png'),
    '--source-reference',(Join-Path $output 'projected_keyframe_031.png'),
    '--output-dir',$output
) 'Soft robe bridge v17 compositing failed'

$reportPath = Join-Path $output 'v17_visual_report.json'
$sceneReceiptPath = Join-Path $output 'scene_receipt.json'
$sceneReceipt = Get-Content -LiteralPath $sceneReceiptPath -Raw | ConvertFrom-Json
$sceneReceipt.classification = 'USER_VISUAL_REVIEW_REQUIRED'
$sceneReceipt.character_variant = 'BLENDER_BODY_PLATE_PLUS_FEATHERED_HEAD_SOFT_ROBE_BRIDGE_V17'
$sceneReceipt.claim_policy = 'The Blender body plate, hidden synthetic head, save/reload, feathered source head and soft robe bridge composites are machine-proven. Final likeness and scene quality remain pending direct visual review.'
$sceneReceipt | Add-Member -NotePropertyName visual_changes_v17 -NotePropertyValue ([ordered]@{
    floating_head_route_rejected = $true
    soft_robe_bridge = $true
    selected_variant = 'bridge185_low'
    animation_rendered = $false
}) -Force
$sceneReceipt | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $sceneReceiptPath -Encoding utf8

$requiredNames = @(
 'beggars_photoreal_recreation.blend','body_plate.png','source_head_cutout.png','projected_keyframe_031.png',
 'hero_bridge_composite.png','contact_sheet.png','scene_receipt.json','v17_visual_report.json',
 'bridge_composite_bridge185_low.png','bridge_composite_bridge185.png','bridge_composite_bridge200.png',
 'bridge_composite_bridge200_low.png','bridge_composite_bridge185_warm.png','bridge_composite_bridge185_narrow.png'
)
foreach ($name in $requiredNames) {
    $path = Join-Path $output $name
    if (-not (Test-Path $path) -or (Get-Item $path).Length -le 0) { throw "V17 output missing: $name" }
}
if ((Get-Item (Join-Path $output 'contact_sheet.png')).Length -lt 100000) { throw 'V17 contact sheet too small' }
if ($env:GITHUB_ENV) { "B17_OUTPUT=$output" | Add-Content $env:GITHUB_ENV }
Write-Host "BLENDER_FACEVERSE_V17_OUTPUT=$output"
Write-Host 'BLENDER_FACEVERSE_V17_RUNNER=PROVEN_USER_VISUAL_REVIEW_REQUIRED'
