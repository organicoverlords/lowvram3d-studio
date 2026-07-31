from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms
from transformers import AutoModelForImageSegmentation

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PACKAGE_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from lowvram3d.avatar_mask import (  # noqa: E402
    decontaminate_edges,
    framing_report,
    normalize_subject,
    refine_alpha,
)

BIREFNET_MODEL = "ZhengPeng7/BiRefNet"
BIREFNET_REVISION = "e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4"


def _extract_logits(output):
    if isinstance(output, (list, tuple)):
        return output[-1]
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, dict):
        for key in ("logits", "preds", "out"):
            if key in output:
                return output[key]
    raise RuntimeError(f"Unsupported BiRefNet output type: {type(output)!r}")


def _biref_mask(image: Image.Image, model_id: str, revision: str, offline: bool) -> tuple[np.ndarray, str]:
    # BiRefNet stays on the GPU, but in fp32 and at 768px. Measured on sm_75 (GTX 1660 SUPER):
    # fp16 returns an all-NaN matte at every resolution and under every SDP backend, and
    # sometimes escalates to "CUDA error: an illegal memory access" which poisons the context.
    # fp32 is correct; 768px peaks at ~2.3 GB against ~3.4 GB for 1024px, which is what pushed
    # total GPU use past the pipeline's VRAM ceiling. Mask coverage is identical to four
    # decimals because the matte is resampled to the full image size below.
    resolution = int(os.environ.get("LOWVRAM3D_BIREFNET_SIZE", "768"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForImageSegmentation.from_pretrained(
        model_id,
        revision=revision,
        trust_remote_code=True,
        local_files_only=offline,
    )
    model.eval().float().to(device)
    transform = transforms.Compose([
        transforms.Resize((resolution, resolution)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    tensor = transform(image.convert("RGB")).unsqueeze(0).float().to(device)
    with torch.inference_mode():
        logits = _extract_logits(model(tensor))
        prediction = logits[0] if logits.ndim >= 3 else logits
        alpha = prediction.squeeze().float().sigmoid().cpu().numpy()
    precision = f"{device}/fp32@{resolution}"
    if not np.isfinite(alpha).all() or float(alpha.max()) <= 0.0:
        raise RuntimeError(f"BiRefNet produced an empty or non-finite matte on {precision}")
    alpha = cv2.resize(alpha, image.size, interpolation=cv2.INTER_CUBIC)
    del tensor, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return np.clip(alpha, 0.0, 1.0), precision


def _pose_data(image: Image.Image) -> tuple[dict, np.ndarray | None]:
    report: dict = {"detected": False, "landmarks": [], "world_landmarks": []}
    try:
        import mediapipe as mp
    except Exception as exc:
        report["error"] = f"mediapipe unavailable: {exc}"
        return report, None

    rgb = np.asarray(image.convert("RGB"))
    with mp.solutions.pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        smooth_landmarks=False,
        enable_segmentation=True,
        min_detection_confidence=0.45,
    ) as pose:
        result = pose.process(rgb)
    if not result.pose_landmarks:
        report["error"] = "No single human pose was detected"
        return report, None

    report["detected"] = True
    report["landmarks"] = [
        {"x": float(point.x), "y": float(point.y), "z": float(point.z), "visibility": float(point.visibility)}
        for point in result.pose_landmarks.landmark
    ]
    if result.pose_world_landmarks:
        report["world_landmarks"] = [
            {"x": float(point.x), "y": float(point.y), "z": float(point.z), "visibility": float(point.visibility)}
            for point in result.pose_world_landmarks.landmark
        ]
    mask = None
    if result.segmentation_mask is not None:
        mask = np.clip(np.asarray(result.segmentation_mask, dtype=np.float32), 0.0, 1.0)
    return report, mask


def _save_preview(image: Image.Image, alpha: np.ndarray, path: Path) -> None:
    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    checker = np.indices((alpha.shape[0], alpha.shape[1])).sum(axis=0) // 24
    checker = np.where(checker[..., None] % 2 == 0, 224.0, 192.0)
    composite = rgb * alpha[..., None] + checker * (1.0 - alpha[..., None])
    Image.fromarray(np.clip(composite, 0, 255).astype(np.uint8), "RGB").save(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask", required=True)
    parser.add_argument("--preview", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--model", default=BIREFNET_MODEL)
    parser.add_argument("--revision", default=BIREFNET_REVISION)
    parser.add_argument("--canvas-size", type=int, default=1024)
    parser.add_argument("--max-input-size", type=int, default=3072)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args()

    opened = ImageOps.exif_transpose(Image.open(args.input))
    original_size = list(opened.size)
    source = opened.convert("RGB")
    max_input = max(1024, min(4096, int(args.max_input_size)))
    if max(source.size) > max_input:
        source.thumbnail((max_input, max_input), Image.Resampling.LANCZOS)
    source_rgb = np.asarray(source, dtype=np.uint8)
    pose, pose_mask = _pose_data(source)
    model_error = ""
    try:
        raw_alpha, device = _biref_mask(source, args.model, args.revision, args.offline)
        backend = "birefnet_pinned"
    except Exception as exc:
        model_error = str(exc)
        if pose_mask is None:
            raise RuntimeError(f"BiRefNet failed and no MediaPipe person mask was available: {exc}") from exc
        raw_alpha = pose_mask
        device = "cpu"
        backend = "mediapipe_segmentation_fallback"

    source_framing = framing_report(raw_alpha, pose)
    alpha, components = refine_alpha(raw_alpha, source_rgb, pose_mask)
    cleaned_rgb = decontaminate_edges(source_rgb, alpha)
    normalized, normalized_alpha, normalized_pose, transform = normalize_subject(
        cleaned_rgb,
        alpha,
        pose,
        canvas_size=max(512, min(2048, int(args.canvas_size))),
        subject_fill=0.86,
    )
    normalized_framing = framing_report(normalized_alpha, normalized_pose)
    if components.get("significant_component_count", 0) > 1 and components.get("second_to_first_ratio", 0.0) > 0.18:
        normalized_framing.setdefault("warnings", []).append(
            "Multiple foreground subjects were detected; only the largest person was kept"
        )
        normalized_framing["ready"] = False

    output = Path(args.output)
    mask_path = Path(args.mask)
    preview = Path(args.preview)
    report_path = Path(args.report)
    for path in (output, mask_path, preview, report_path):
        path.parent.mkdir(parents=True, exist_ok=True)

    normalized.save(output)
    mask_image = Image.fromarray(np.round(normalized_alpha * 255.0).astype(np.uint8), "L")
    mask_image.save(mask_path)
    _save_preview(normalized.convert("RGB"), normalized_alpha, preview)

    report = {
        "success": True,
        "backend": backend,
        "model": args.model,
        "model_revision": args.revision,
        "device": device,
        "original_input_size": original_size,
        "working_input_size": list(source.size),
        "output_size": list(normalized.size),
        "output": str(output),
        "mask": str(mask_path),
        "model_error": model_error,
        "pose": normalized_pose,
        "source_framing": source_framing,
        "framing": normalized_framing,
        "components": components,
        "normalization": transform,
        "alpha": {
            "transparent_fraction": round(float(np.mean(normalized_alpha < 0.02)), 6),
            "opaque_fraction": round(float(np.mean(normalized_alpha > 0.98)), 6),
            "mean": round(float(normalized_alpha.mean()), 6),
        },
        "edge_treatment": {
            "guided_filter": True,
            "foreground_color_decontamination": True,
            "largest_subject_only": True,
        },
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({"success": True, "backend": backend, "framing": normalized_framing}, indent=2))


if __name__ == "__main__":
    main()
