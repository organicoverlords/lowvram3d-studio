"""Extract aggregate low-frequency material statistics from direct observations only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, np.uint8).reshape(-1, 1, 3)
    return cv2.cvtColor(values, cv2.COLOR_RGB2LAB).reshape(-1, 3).astype(np.float32)


def extract(triangle_rgb: np.ndarray, direct: np.ndarray, regions: np.ndarray) -> dict:
    triangle_rgb = np.asarray(triangle_rgb, np.uint8)
    direct = np.asarray(direct, bool)
    regions = np.asarray(regions, np.int32)
    if triangle_rgb.shape != (len(direct), 3) or regions.shape != direct.shape:
        raise ValueError("triangle RGB, direct mask, and regions must align")
    palettes = {}
    for region in np.unique(regions[direct]):
        values = triangle_rgb[direct & (regions == region)]
        if not len(values):
            continue
        lab = rgb_to_lab(values)
        med = np.median(lab, axis=0)
        palettes[str(int(region))] = {
            "region": int(region), "sample_count": int(len(values)),
            "median_lab": [float(v) for v in med],
            "p05_lab": [float(v) for v in np.percentile(lab, 5, axis=0)],
            "p95_lab": [float(v) for v in np.percentile(lab, 95, axis=0)],
            "authority": "LOW_ONLY", "stores_patches": False,
        }
    values = triangle_rgb[direct]
    if len(values):
        lab = rgb_to_lab(values)
        palettes["global"] = {"region": -1, "sample_count": int(len(values)),
                               "median_lab": [float(v) for v in np.median(lab, axis=0)],
                               "authority": "LOW_ONLY", "stores_patches": False}
    return {"schema": "material_palettes_v1", "palettes": palettes,
            "direct_observation_only": True, "high_frequency_removed": True}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--triangle-rgb", required=True)
    parser.add_argument("--direct-mask", required=True)
    parser.add_argument("--regions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = extract(np.load(args.triangle_rgb), np.load(args.direct_mask), np.load(args.regions))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(data, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
