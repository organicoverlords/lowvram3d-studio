[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$target = 'tools\beggars_scene\run_faceverse_v4_preflight.py'
if (-not (Test-Path -LiteralPath $target)) {
    throw "FaceVerse proof source is missing: $target"
}
$text = Get-Content -LiteralPath $target -Raw

if (-not $text.Contains('def faceverse_stage(name: str) -> None:')) {
    $text = $text.Replace(
        "import argparse`n",
        "import argparse`nimport faulthandler`nimport traceback`n"
    )
    $anchor = "import torch`n`n"
    $helper = @'
import torch


def faceverse_stage(name: str) -> None:
    print(f"FACEVERSE_STAGE={name}", flush=True)

'@
    if (-not $text.Contains($anchor)) {
        throw 'Could not locate the FaceVerse torch import anchor.'
    }
    $text = $text.Replace($anchor, $helper)

    $text = $text.Replace(
        "def main() -> int:`n    args = parse_args()",
        "def main() -> int:`n    faulthandler.enable()`n    faceverse_stage(\"MAIN_BEGIN\")`n    args = parse_args()`n    faceverse_stage(\"ARGS_PARSED\")"
    )
    $text = $text.Replace(
        "    sys.path.insert(0, str(faceverse_root))`n    from faceversev4 import FaceVerseRecon",
        "    sys.path.insert(0, str(faceverse_root))`n    faceverse_stage(\"FACEVERSE_IMPORT_BEGIN\")`n    from faceversev4 import FaceVerseRecon"
    )
    $text = $text.Replace(
        "    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel`n`n    image_bgr = cv2.imread",
        "    from Sim3DR.renderer import render_fvr  # pylint: disable=import-error,import-outside-toplevel`n    faceverse_stage(\"FACEVERSE_IMPORT_DONE\")`n`n    faceverse_stage(\"IMAGE_READ_BEGIN\")`n    image_bgr = cv2.imread"
    )
    $text = $text.Replace(
        "    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)`n    bbox, eye_coefficients = detect_face_box_and_eyes(image_rgb, landmarker_path)",
        "    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)`n    faceverse_stage(\"IMAGE_READ_DONE\")`n    faceverse_stage(\"MEDIAPIPE_BEGIN\")`n    bbox, eye_coefficients = detect_face_box_and_eyes(image_rgb, landmarker_path)`n    faceverse_stage(\"MEDIAPIPE_DONE\")"
    )
    $text = $text.Replace(
        "    device = choose_device(args.device)`n",
        "    device = choose_device(args.device)`n    print(f\"FACEVERSE_DEVICE={device}\", flush=True)`n"
    )
    $text = $text.Replace(
        "    start = time.perf_counter()`n    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)`n    load_seconds = time.perf_counter() - start",
        "    faceverse_stage(\"MODEL_LOAD_BEGIN\")`n    start = time.perf_counter()`n    model = FaceVerseRecon(str(model_path), str(checkpoint_path), device)`n    load_seconds = time.perf_counter() - start`n    print(f\"FACEVERSE_MODEL_LOAD_SECONDS={load_seconds:.6f}\", flush=True)`n    faceverse_stage(\"MODEL_LOAD_DONE\")"
    )
    $text = $text.Replace(
        "    inference_start = time.perf_counter()`n    coefficients, bbox_list = model.process_imgs(",
        "    faceverse_stage(\"INFERENCE_BEGIN\")`n    inference_start = time.perf_counter()`n    coefficients, bbox_list = model.process_imgs("
    )
    $text = $text.Replace(
        "    inference_seconds = time.perf_counter() - inference_start`n`n    triangles =",
        "    inference_seconds = time.perf_counter() - inference_start`n    print(f\"FACEVERSE_INFERENCE_SECONDS={inference_seconds:.6f}\", flush=True)`n    faceverse_stage(\"INFERENCE_DONE\")`n`n    faceverse_stage(\"RENDER_BEGIN\")`n    triangles ="
    )
    $text = $text.Replace(
        "    if render_rgb.shape[:2] != image_rgb.shape[:2]:",
        "    faceverse_stage(\"RENDER_DONE\")`n    if render_rgb.shape[:2] != image_rgb.shape[:2]:"
    )
    $text = $text.Replace(
        "    if not cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)):",
        "    faceverse_stage(\"OUTPUT_WRITE_BEGIN\")`n    if not cv2.imwrite(str(render_path), cv2.cvtColor(render_rgb, cv2.COLOR_RGB2BGR)):"
    )
    $text = $text.Replace(
        "    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=\"utf-8\")",
        "    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding=\"utf-8\")`n    faceverse_stage(\"OUTPUT_WRITE_DONE\")"
    )
    $oldBottom = @'
if __name__ == "__main__":
    raise SystemExit(main())
'@
    $newBottom = @'
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
    if (-not $text.Contains($oldBottom)) {
        throw 'Could not locate the FaceVerse proof main boundary.'
    }
    $text = $text.Replace($oldBottom, $newBottom)
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
