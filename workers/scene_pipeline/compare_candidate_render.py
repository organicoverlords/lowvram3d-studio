"""Compare exact-source candidate renders against the normalized source."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(r"C:\AI\ScenePipelineSmoke\20260803\castlegrounds")
PROOF = Path(__file__).resolve().parents[2] / "proof" / "scene" / "20260803-image-to-scene-smoke"


def metrics(prefix: str) -> dict[str, object]:
    source = cv2.cvtColor(cv2.imread(str(ROOT / "source_rgb_512.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    mask = np.load(ROOT / "mask.npy").astype(bool) & np.isfinite(np.load(ROOT / "points.npy")).all(axis=-1)
    off = cv2.cvtColor(cv2.imread(str(ROOT / f"{prefix}_cull_off.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    on = cv2.cvtColor(cv2.imread(str(ROOT / f"{prefix}_cull_on.png"), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32)
    off_fg = np.max(off, axis=2) > 8
    on_fg = np.max(on, axis=2) > 8
    target = np.max(source, axis=2) > 8
    inside = mask
    a = source[inside].reshape(-1, 3)
    b = off[inside].reshape(-1, 3)
    aa = a - a.mean(axis=0, keepdims=True)
    bb = b - b.mean(axis=0, keepdims=True)
    ncc = float(np.sum(aa * bb) / max(np.sqrt(np.sum(aa * aa) * np.sum(bb * bb)), 1e-8))
    diff = a - b
    psnr = float(20.0 * np.log10(255.0 / max(float(np.sqrt(np.mean(diff * diff))), 1e-8)))
    edge_a = cv2.Canny(source.astype(np.uint8), 80, 160).astype(np.float32)
    edge_b = cv2.Canny(off.astype(np.uint8), 80, 160).astype(np.float32)
    edge_a -= edge_a.mean(); edge_b -= edge_b.mean()
    edge_corr = float(np.sum(edge_a * edge_b) / max(np.sqrt(np.sum(edge_a * edge_a) * np.sum(edge_b * edge_b)), 1e-8))
    intersection = np.count_nonzero(off_fg & target & inside)
    union = np.count_nonzero((off_fg | target) & inside)
    off_cov = float(np.count_nonzero(off_fg & inside) / max(np.count_nonzero(inside), 1))
    on_cov = float(np.count_nonzero(on_fg & inside) / max(np.count_nonzero(inside), 1))
    receipt = {
        "schema": "castlegrounds_exact_source_candidate_comparison_v1",
        "prefix": prefix,
        "renders": {"cull_off": str(ROOT / f"{prefix}_cull_off.png"), "cull_on": str(ROOT / f"{prefix}_cull_on.png")},
        "valid_mask_coverage": float(mask.mean()),
        "rendered_nonblack_coverage_inside_mask": off_cov,
        "cull_on_coverage_inside_mask": on_cov,
        "cull_on_cull_off_delta": abs(on_cov - off_cov),
        "silhouette_iou_inside_mask": float(intersection / max(union, 1)),
        "ssim_proxy_ncc": ncc,
        "psnr_db": psnr,
        "mean_abs_colour_error": float(np.mean(np.abs(diff))),
        "edge_correlation": edge_corr,
        "cull_on_within_one_percent": bool(abs(on_cov - off_cov) <= 0.01),
    }
    (ROOT / f"{prefix}_comparison.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    (PROOF / f"{prefix}_comparison.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: python compare_candidate_render.py PREFIX")
    print(json.dumps(metrics(sys.argv[1]), indent=2))
