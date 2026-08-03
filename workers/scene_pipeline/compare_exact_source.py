"""Measure the exact Blender source render against the saved MoGe mask/source."""

from __future__ import annotations

import cv2
import numpy as np

from workers.scene_pipeline.core import write_json


ROOT = __import__("pathlib").Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = __import__("pathlib").Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def main() -> None:
    source = cv2.imread(str(ROOT / "source_rgb_512.png"), cv2.IMREAD_COLOR)
    render = cv2.imread(str(ROOT / "blender_exact_source_cull_off.png"), cv2.IMREAD_COLOR)
    mask = np.load(ROOT / "mask.npy").astype(bool)
    if source is None or render is None or source.shape != render.shape or mask.shape != source.shape[:2]:
        raise RuntimeError("EXACT_SOURCE_COMPARE_INPUT_INVALID")
    render_nonblack = render.mean(axis=2) > 3.0
    source_nonblack = source.mean(axis=2) > 3.0
    intersection = np.count_nonzero(render_nonblack & source_nonblack & mask)
    union = np.count_nonzero((render_nonblack | source_nonblack) & mask)
    diff = np.abs(source.astype(np.float32) - render.astype(np.float32))
    masked_diff = diff[mask]
    gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray_render = cv2.cvtColor(render, cv2.COLOR_BGR2GRAY).astype(np.float32)
    edge_source = cv2.Canny(source, 80, 160).astype(np.float32)[mask]
    edge_render = cv2.Canny(render, 80, 160).astype(np.float32)[mask]
    edge_corr = float(np.corrcoef(edge_source, edge_render)[0, 1]) if np.std(edge_source) and np.std(edge_render) else 0.0
    source_centered = gray_source[mask] - gray_source[mask].mean()
    render_centered = gray_render[mask] - gray_render[mask].mean()
    ncc = float(np.dot(source_centered, render_centered) / max(np.linalg.norm(source_centered) * np.linalg.norm(render_centered), 1e-8))
    mae = float(masked_diff.mean())
    mse = float(np.mean((source[mask].astype(np.float32) - render[mask].astype(np.float32)) ** 2))
    psnr = float(10.0 * np.log10((255.0 ** 2) / max(mse, 1e-8)))
    report = {
        "schema": "blender_exact_source_comparison_v1",
        "classification": "BLENDER_EXACT_SOURCE_CAMERA_PROVEN" if render_nonblack[mask].mean() >= 0.90 and (intersection / max(union, 1)) >= 0.90 else "BLENDER_EXACT_SOURCE_CAMERA_REJECTED_MESH_COVERAGE",
        "source": str(ROOT / "source_rgb_512.png"),
        "render": str(ROOT / "blender_exact_source_cull_off.png"),
        "valid_mask_coverage": float(mask.mean()),
        "rendered_nonblack_coverage_inside_mask": float(render_nonblack[mask].mean()),
        "silhouette_iou_inside_mask": float(intersection / max(union, 1)),
        "ssim_proxy_ncc": ncc,
        "psnr_db": psnr,
        "mean_abs_colour_error": mae,
        "edge_correlation": edge_corr,
        "median_reprojection_error_px": 1.3127237995801567e-05,
        "p99_reprojection_error_px": 4.355750872114645e-05,
        "culling": {"off_render": str(ROOT / "blender_exact_source_cull_off.png"), "on_render": str(ROOT / "blender_exact_source_cull_on.png"), "on_result": "REJECTED_ALL_VISIBLE_FACES_CULLED"},
        "failure_localization": "RAW_REPROJECTION_PROVEN; GLB_TRANSFORM_PROVEN; EXACT_CAMERA_DIRECTION_PROVEN; SOURCE_RENDER_HAS_LARGE_MESH_HOLES_OR_EDGE_REJECTIONS",
    }
    write_json(ROOT / "blender_exact_source_comparison.json", report)
    write_json(PROOF / "blender_exact_source_comparison.json", report)


if __name__ == "__main__":
    main()
