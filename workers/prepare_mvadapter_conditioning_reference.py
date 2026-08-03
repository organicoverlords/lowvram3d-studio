"""Prepare the official-style MV-Adapter appearance conditioning image.

This is deliberately independent from CPU projection registration.  It uses
only source-image alpha (or a deterministic border-colour matte for opaque
inputs), preserves the subject aspect ratio, and composites the outside to
neutral gray.  It never compares against a mesh silhouette.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _alpha_from_image(image: np.ndarray) -> tuple[np.ndarray, str]:
    if image.ndim == 3 and image.shape[2] == 4:
        alpha = image[:, :, 3]
        if np.count_nonzero(alpha) > 0 and np.count_nonzero(alpha < 250) > max(8, alpha.size * 0.001):
            return alpha, "source_alpha"
    rgb = image[:, :, :3].astype(np.float32)
    border = np.concatenate([rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]], axis=0)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb - background, axis=2)
    threshold = max(12.0, float(np.percentile(distance, 55)))
    mask = (distance > threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    return mask, "deterministic_border_colour_matte"


def prepare(source: Path, output: Path, report_path: Path, size: int = 256) -> dict[str, Any]:
    if size < 32:
        raise RuntimeError("MVADAPTER_CONDITIONING_SIZE_INVALID")
    image = cv2.imread(str(source), cv2.IMREAD_UNCHANGED)
    if image is None or image.ndim not in (2, 3):
        raise RuntimeError("MVADAPTER_CONDITIONING_SOURCE_UNREADABLE")
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGRA)
    if image.shape[2] == 3:
        image = np.dstack([image, np.full(image.shape[:2], 255, np.uint8)])
    alpha, alpha_source = _alpha_from_image(image)
    foreground = alpha > 32
    ys, xs = np.nonzero(foreground)
    if len(xs) == 0:
        raise RuntimeError("MVADAPTER_CONDITIONING_ALPHA_EMPTY")
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    crop = image[y0:y1, x0:x1]
    crop_h, crop_w = crop.shape[:2]
    if min(crop_h, crop_w) <= 0:
        raise RuntimeError("MVADAPTER_CONDITIONING_CROP_INVALID")
    longer = max(crop_h, crop_w)
    target_longer = int(round(size * 0.90))
    if crop_h >= crop_w:
        resized_h = target_longer
        resized_w = max(1, int(round(crop_w * target_longer / crop_h)))
    else:
        resized_w = target_longer
        resized_h = max(1, int(round(crop_h * target_longer / crop_w)))
    resized = cv2.resize(crop, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((size, size, 4), np.uint8)
    offset_x = (size - resized_w) // 2
    offset_y = (size - resized_h) // 2
    if offset_x < 0 or offset_y < 0 or offset_x + resized_w > size or offset_y + resized_h > size:
        raise RuntimeError("MVADAPTER_CONDITIONING_PLACEMENT_CLIPPED")
    canvas[offset_y : offset_y + resized_h, offset_x : offset_x + resized_w] = resized
    outside = canvas[:, :, 3] <= 32
    canvas[outside, :3] = 127
    canvas[outside, 3] = 0
    occupancy = max(resized_w, resized_h) / float(size)
    if not 0.88 <= occupancy <= 0.92:
        raise RuntimeError(f"MVADAPTER_CONDITIONING_OCCUPANCY_INVALID:{occupancy:.6f}")
    if not np.all(canvas[0, 0, :3] == 127):
        raise RuntimeError("MVADAPTER_CONDITIONING_BACKGROUND_INVALID")
    output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output), canvas):
        raise RuntimeError("MVADAPTER_CONDITIONING_WRITE_FAILED")
    report = {
        "schema": "lowvram3d_mvadapter_conditioning_reference_v1",
        "source": str(source),
        "source_sha256": sha256(source),
        "source_dimensions": [int(image.shape[1]), int(image.shape[0])],
        "alpha_source": alpha_source,
        "source_alpha_bounds": [x0, y0, x1, y1],
        "output": str(output),
        "output_sha256": sha256(output),
        "output_dimensions": [size, size],
        "resized_subject_dimensions": [resized_w, resized_h],
        "occupancy_fraction": round(float(occupancy), 6),
        "placement_offsets": [offset_x, offset_y],
        "background_rgb": [127, 127, 127],
        "mirror_used": False,
        "non_rigid_warp_used": False,
        "mesh_silhouette_comparison_used": False,
        "aspect_ratio_preserved": True,
        "deterministic": True,
        "passed": True,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    report = prepare(Path(args.source), Path(args.output), Path(args.report), args.size)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
