"""Compute objective landmark parallax from MoGe depth and intrinsics."""

from __future__ import annotations

import numpy as np
from pathlib import Path

from workers.scene_pipeline.core import write_json


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def project(points: np.ndarray, intrinsics: np.ndarray, translation: np.ndarray, width: int, height: int) -> np.ndarray:
    fx, fy, cx, cy = intrinsics[0, 0] * width, intrinsics[1, 1] * height, intrinsics[0, 2] * width, intrinsics[1, 2] * height
    shifted = points - translation[None, :]
    return np.stack((fx * shifted[:, 0] / np.maximum(shifted[:, 2], 1e-6) + cx, fy * shifted[:, 1] / np.maximum(shifted[:, 2], 1e-6) + cy), axis=1)


def main() -> None:
    points = np.load(ROOT / "points.npy")
    depth = np.load(ROOT / "depth.npy")
    mask = np.load(ROOT / "mask.npy").astype(bool)
    intrinsics = np.load(ROOT / "intrinsics.npy")
    height, width = depth.shape
    coords = np.argwhere(mask)
    values = depth[mask]
    valid_points = points[mask]
    order = np.argsort(values)
    band_size = max(20, len(order) // 20)
    middle_size = max(10, len(order) // 40)
    bands = {
        "near": order[:band_size],
        "middle": order[len(order) // 2 - middle_size: len(order) // 2 + middle_size],
        "far": order[-band_size:],
    }
    median_depth = float(np.median(values))
    lateral = 0.02 * median_depth
    forward = 0.01 * median_depth
    view_results = {}
    translations = {
        "left": np.array([-lateral, 0.0, 0.0]),
        "right": np.array([lateral, 0.0, 0.0]),
        "forward": np.array([0.0, 0.0, -forward]),
        "elevated": np.array([0.0, -0.02 * median_depth, 0.02 * median_depth]),
    }
    for name, translation in translations.items():
        per_band = {}
        for band, indices in bands.items():
            source = coords[indices][:, ::-1].astype(np.float32)
            moved = project(valid_points[indices], intrinsics, translation, width, height)
            displacement = np.linalg.norm(moved - source, axis=1)
            per_band[band] = {
                "count": int(len(displacement)),
                "median_pixels": float(np.median(displacement)),
                "p95_pixels": float(np.percentile(displacement, 95)),
            }
        view_results[name] = {"camera_displacement": translation.tolist(), "bands": per_band}
    near = np.median([view_results[name]["bands"]["near"]["median_pixels"] for name in ("left", "right")])
    far = np.median([view_results[name]["bands"]["far"]["median_pixels"] for name in ("left", "right")])
    ratio = float(near / max(far, 1e-6))
    receipt = {
        "schema": "scene_parallax_receipt_v1",
        "classification": "PARALLAX_PROVEN" if near >= 3.0 and ratio >= 1.5 else "PARALLAX_REJECTED_FLAT_OR_INVALID",
        "source_resolution": [width, height],
        "median_valid_depth": median_depth,
        "lateral_baseline": lateral,
        "forward_baseline": forward,
        "landmark_bands": view_results,
        "near_median_lateral_displacement_px": float(near),
        "far_median_lateral_displacement_px": float(far),
        "near_far_displacement_ratio": ratio,
        "newly_revealed_geometry": "REQUIRES_RENDER_OCCLUSION_COMPARISON",
    }
    write_json(PROOF / "parallax_receipt.json", receipt)
    write_json(ROOT / "parallax_receipt.json", receipt)


if __name__ == "__main__":
    main()
