"""Register the real panda source to the CPU front control silhouette.

Only bounded similarity/affine box fitting is used.  No background-removal,
mirroring, non-rigid warp, or learned image processing is allowed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any
import sys

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from shaman_texture_views import mask_iou, refine_box, subject_bbox, warp_to_frame
from lowvram3d.asset_profiles import foreground_mask


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def register(source: Path, target_mask_path: Path, output: Path, report_path: Path, size: int = 256) -> dict[str, Any]:
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    target = cv2.imread(str(target_mask_path), cv2.IMREAD_UNCHANGED)
    if image is None or target is None:
        raise RuntimeError("REGISTERED_REFERENCE_INPUT_MISSING")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    target_mask = target[:, :, 3] > 32 if target.ndim == 3 and target.shape[2] == 4 else target > 32
    target_mask = cv2.resize(target_mask.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST) > 0
    source_mask = foreground_mask(image).astype(bool)
    if not source_mask.any() or not target_mask.any():
        raise RuntimeError("REGISTERED_REFERENCE_EMPTY_MASK")
    source_box = subject_bbox(source_mask)
    target_box = subject_bbox(target_mask)
    fitted_box, iou = refine_box(source_mask, target_mask, source_box, target_box, size)
    rgb = image[:, :, :3]
    alpha = image[:, :, 3]
    registered_rgb = warp_to_frame(rgb, source_box, fitted_box, size, interpolation=cv2.INTER_LINEAR)
    registered_alpha = warp_to_frame(alpha, source_box, fitted_box, size, interpolation=cv2.INTER_NEAREST)
    registered_mask = registered_alpha > 32
    if float(iou) < 0.85:
        failure_report = {
            "schema": "lowvram3d_mvadapter_registered_reference_v1",
            "source": str(source),
            "source_sha256": sha256(source),
            "target_mask": str(target_mask_path),
            "target_mask_sha256": sha256(target_mask_path),
            "transform_type": "bounded_similarity_or_affine_bbox",
            "silhouette_iou": round(float(mask_iou(registered_mask, target_mask)), 6),
            "required_silhouette_iou": 0.85,
            "mirror_used": False,
            "non_rigid_warp_used": False,
            "passed": False,
            "failure": "REGISTERED_REFERENCE_IOU_BELOW_THRESHOLD",
        }
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(failure_report, indent=2), encoding="utf-8")
        raise RuntimeError(f"REGISTERED_REFERENCE_IOU_BELOW_THRESHOLD:{float(iou):.6f}")
    registered_rgb = np.where(
        registered_mask[..., None], registered_rgb, np.full_like(registered_rgb, 127)
    )
    rgba = np.dstack([registered_rgb, registered_alpha]).astype(np.uint8)
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), rgba):
        raise RuntimeError("REGISTERED_REFERENCE_WRITE_FAILED")
    report = {
        "schema": "lowvram3d_mvadapter_registered_reference_v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "registered": str(output),
        "registered_sha256": sha256(output),
        "target_mask": str(target_mask_path),
        "target_mask_sha256": sha256(target_mask_path),
        "transform_type": "bounded_similarity_or_affine_bbox",
        "transform_matrix": "recorded_by_box_contract",
        "silhouette_iou": round(float(mask_iou(registered_mask, target_mask)), 6),
        "fitted_box": list(fitted_box),
        "crop_bounds": list(source_box),
        "output_dimensions": [size, size],
        "mirror_used": False,
        "non_rigid_warp_used": False,
        "background": "neutral_mid_gray_0.5_outside_registered_alpha",
        "passed": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--target-mask", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    report = register(Path(args.source), Path(args.target_mask), Path(args.output), Path(args.report), args.size)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
