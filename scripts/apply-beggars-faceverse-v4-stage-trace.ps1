[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'tools\beggars_scene\run_faceverse_v4_preflight.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse proof source is missing: $target"
}
$text = (Get-Content -LiteralPath $target -Raw).Replace("`r`n", "`n")

function Replace-Required {
    param(
        [Parameter(Mandatory)][string]$Label,
        [Parameter(Mandatory)][string]$Old,
        [Parameter(Mandatory)][string]$New
    )
    if (-not $script:text.Contains($Old)) {
        throw "Could not locate FaceVerse stage-trace anchor: $Label"
    }
    $script:text = $script:text.Replace($Old, $New)
}

if (-not $text.Contains('def faceverse_stage(name: str) -> None:')) {
    Replace-Required -Label 'imports' -Old @'
import argparse
import hashlib
'@ -New @'
import argparse
import faulthandler
import hashlib
import traceback
'@

    Replace-Required -Label 'stage helper' -Old @'
import torch


def parse_args()
'@ -New @'
import torch


def faceverse_stage(name: str) -> None:
    print(f"FACEVERSE_STAGE={name}", flush=True)


def parse_args()
'@

    Replace-Required -Label 'main entry' -Old @'
def main() -> int:
    args = parse_args()
'@ -New @'
def main() -> int:
    faulthandler.enable()
    faceverse_stage("MAIN_BEGIN")
    args = parse_args()
    faceverse_stage("ARGS_PARSED")
'@

    Replace-Required -Label 'FaceVerse import begin' -Old @'
    sys.path.insert(0, str(faceverse_root))
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
'@ -New @'
    sys.path.insert(0, str(faceverse_root))
    faceverse_stage("FACEVERSE_IMPORT_BEGIN")
    from faceversev4 import FaceVerseRecon  # pylint: disable=import-error,import-outside-toplevel
'@

    Replace-Required -Label 'FaceVerse import done' -Old @'
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel

    image_bgr = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
'@ -New @'
    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel
    faceverse_stage("FACEVERSE_IMPORT_DONE")

    faceverse_stage("IMAGE_READ_BEGIN")
    image_bgr = cv2.imread(str(input_image_path), cv2.IMREAD_COLOR)
'@

    Replace-Required -Label 'MediaPipe stages' -Old @'
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    bbox, eye_coefficients = detect_face_box_and_eyes(image_rgb, landmarker_path)

    device = choose_device(args.device)
'@ -New @'
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    faceverse_stage("IMAGE_READ_DONE")
    faceverse_stage("MEDIAPIPE_BEGIN")
    bbox, eye_coefficients = detect_face_box_and_eyes(image_rgb, landmarker_path)
    faceverse_stage("MEDIAPIPE_DONE")

    device = choose_device(args.device)
    print(f"FACEVERSE_DEVICE={device}", flush=True)
'@

    Replace-Required -Label 'model load stages' -Old @'
    start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    load_seconds = time.perf_counter() - start

    inference_start = time.perf_counter()
'@ -New @'
    faceverse_stage("MODEL_LOAD_BEGIN")
    start = time.perf_counter()
    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)
    load_seconds = time.perf_counter() - start
    print(f"FACEVERSE_MODEL_LOAD_SECONDS={load_seconds:.6f}", flush=True)
    faceverse_stage("MODEL_LOAD_DONE")

    faceverse_stage("INFERENCE_BEGIN")
    inference_start = time.perf_counter()
'@

    Replace-Required -Label 'inference and render stages' -Old @'
    inference_seconds = time.perf_counter() - inference_start

    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)
'@ -New @'
    inference_seconds = time.perf_counter() - inference_start
    print(f"FACEVERSE_INFERENCE_SECONDS={inference_seconds:.6f}", flush=True)
    faceverse_stage("INFERENCE_DONE")

    faceverse_stage("RENDER_BEGIN")
    triangles = np.asarray(model.fvd["tri"], dtype=np.int32)
'@

    Replace-Required -Label 'render done' -Old @'
    if render_rgb.shape[:2] != image_rgb.shape[:2]:
'@ -New @'
    faceverse_stage("RENDER_DONE")
    if render_rgb.shape[:2] != image_rgb.shape[:2]:
'@

    Replace-Required -Label 'output write begin' -Old @'
    if not cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)):
'@ -New @'
    faceverse_stage("OUTPUT_WRITE_BEGIN")
    if not cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)):
'@

    Replace-Required -Label 'output write done' -Old @'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    if coefficients.shape[1] != 621:
'@ -New @'
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    faceverse_stage("OUTPUT_WRITE_DONE")

    if coefficients.shape[1] != 621:
'@

    Replace-Required -Label 'main exception boundary' -Old @'
if __name__ == "__main__":
    raise SystemExit(main())
'@ -New @'
if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except BaseException:  # diagnostic boundary
        traceback.print_exc()
        faceverse_stage("UNHANDLED_EXCEPTION")
        raise
'@
}
else {
    Write-Host 'FACEVERSE_STAGE_TRACE=ALREADY_APPLIED'
}

[System.IO.File]::WriteAllText((Resolve-Path -LiteralPath $target), $text, [System.Text.UTF8Encoding]::new($false))

$python = 'C:\AI\LowVRAM3D-cache\faceverse-v4\venv-py39-cu118\Scripts\python.exe'
if (Test-Path -LiteralPath $python) {
    & $python -m py_compile $target
    if ($LASTEXITCODE -ne 0) {
        throw 'FaceVerse stage-traced proof source failed compilation.'
    }
}
Write-Host 'FACEVERSE_STAGE_TRACE=PROVEN'
