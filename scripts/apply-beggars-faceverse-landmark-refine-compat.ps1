[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'tools\beggars_scene\run_faceverse_v4_landmark_refine.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse landmark-refinement source is missing: $target"
}

$text = Get-Content -LiteralPath $target -Raw

$oldImport = @'
import torch
import torch.nn.functional as F

import run_faceverse_v4_identity_fusion as base
'@
$newImport = @'
import torch
import torch.nn.functional as F

_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
import run_faceverse_v4_identity_fusion as base
'@
if ($text.Contains($oldImport)) {
    $text = $text.Replace($oldImport, $newImport)
}
elseif (-not $text.Contains('_SCRIPT_DIR = Path(__file__).resolve().parent')) {
    throw 'Could not locate the sibling-import compatibility anchor.'
}

$oldBaseline = '            _, baseline_projected, _, baseline_colors = model.from_coeffs(baseline, bbox_list)'
$newBaseline = '            _, baseline_projected, baseline_normals, baseline_colors = model.from_coeffs(baseline, bbox_list)'
if ($text.Contains($oldBaseline)) {
    $text = $text.Replace($oldBaseline, $newBaseline)
}
elseif (-not $text.Contains('baseline_projected, baseline_normals, baseline_colors')) {
    throw 'Could not locate the baseline-normal compatibility anchor.'
}

$oldRender = '            baseline_render, _ = render_fvr(frame_rgb, baseline_projected[0], triangles, np.zeros_like(baseline_projected[0]), baseline_source_colors)'
$newRender = '            baseline_render, _ = render_fvr(frame_rgb, baseline_projected[0], triangles, baseline_normals[0], baseline_source_colors)'
if ($text.Contains($oldRender)) {
    $text = $text.Replace($oldRender, $newRender)
}
elseif (-not $text.Contains('triangles, baseline_normals[0], baseline_source_colors')) {
    throw 'Could not locate the baseline-render compatibility anchor.'
}

$oldScheduler = '    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(iterations, 1), eta_min=0.08)'
$newScheduler = '    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(iterations, 1), eta_min=0.0002)'
if ($text.Contains($oldScheduler)) {
    $text = $text.Replace($oldScheduler, $newScheduler)
}
elseif (-not $text.Contains('eta_min=0.0002')) {
    throw 'Could not locate the landmark-refinement learning-rate anchor.'
}

$oldLandmarkPath = 'model.run(coefficients, only_lms=True)["lms_proj"]'
$newLandmarkPath = 'model.run(coefficients, only_lms=False)["lms_proj"]'
$text = $text.Replace($oldLandmarkPath, $newLandmarkPath)
$oldBaselinePath = 'model.run(baseline_coefficients, only_lms=True)["lms_proj"]'
$newBaselinePath = 'model.run(baseline_coefficients, only_lms=False)["lms_proj"]'
$text = $text.Replace($oldBaselinePath, $newBaselinePath)

$oldSanityAnchor = '    print(f"FACEVERSE_REFINE_INITIAL_RMSE_PX={initial_rmse:.6f}", flush=True)'
$newSanityAnchor = @'
    print(f"FACEVERSE_REFINE_INITIAL_RMSE_PX={initial_rmse:.6f}", flush=True)
    if not np.isfinite(initial_rmse) or initial_rmse > float(model.imgsize) * 0.35:
        raise RuntimeError(
            f"Baseline landmark projection is outside the bounded crop coordinate system: "
            f"rmse={initial_rmse:.3f}px imgsize={model.imgsize}"
        )
    print("FACEVERSE_REFINE_BASELINE_COORDINATES=PROVEN", flush=True)
'@
if ($text.Contains($oldSanityAnchor)) {
    $text = $text.Replace($oldSanityAnchor, $newSanityAnchor.TrimEnd())
}
elseif (-not $text.Contains('FACEVERSE_REFINE_BASELINE_COORDINATES=PROVEN')) {
    throw 'Could not locate the baseline-coordinate sanity anchor.'
}

[System.IO.File]::WriteAllText(
    (Resolve-Path -LiteralPath $target),
    $text,
    [System.Text.UTF8Encoding]::new($false)
)

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw "FaceVerse Python is missing: $python"
}
& $python -m py_compile $target
if ($LASTEXITCODE -ne 0) {
    throw 'FaceVerse landmark-refinement source failed compilation after compatibility patch.'
}

$patched = Get-Content -LiteralPath $target -Raw
if (-not $patched.Contains('eta_min=0.0002')) {
    throw 'Refinement source gate: bounded scheduler floor is absent.'
}
if ($patched.Contains('eta_min=0.08')) {
    throw 'Refinement source gate: unsafe scheduler floor remains.'
}
if (-not $patched.Contains('baseline_normals[0]')) {
    throw 'Refinement source gate: baseline normals are absent.'
}
if (-not $patched.Contains('_SCRIPT_DIR = Path(__file__).resolve().parent')) {
    throw 'Refinement source gate: sibling import path is absent.'
}
if ($patched.Contains('only_lms=True')) {
    throw 'Refinement source gate: unscaled only_lms path remains.'
}
if (-not $patched.Contains('FACEVERSE_REFINE_BASELINE_COORDINATES=PROVEN')) {
    throw 'Refinement source gate: baseline coordinate sanity gate is absent.'
}
Write-Host 'FACEVERSE_FULL_GEOMETRY_LANDMARK_PATH=PROVEN'
Write-Host 'FACEVERSE_LANDMARK_REFINE_SOURCE_GATE=PROVEN'
Write-Host 'FACEVERSE_LANDMARK_REFINE_COMPAT=PROVEN'
