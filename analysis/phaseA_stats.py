"""PHASE A statistics: per-view luminance over model pixels only, for each diagnostic material."""
from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

DIAG = sys.argv[1]
ATLAS = sys.argv[2]
COVERAGE = sys.argv[3]
OUT = sys.argv[4]

VIEWS = ["front", "back", "left", "right", "top", "underside"]
MODES = ["BASECOLOR_EMISSION", "NEUTRAL_CLAY", "CURRENT_MATERIAL"]


def luminance(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


report: dict = {"views": {}, "modes": {}}
for mode in MODES:
    per_view = {}
    for view in VIEWS:
        path = os.path.join(DIAG, f"{mode}_{view}.png")
        image = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if image is None:
            continue
        if image.shape[2] == 4:
            alpha = image[..., 3].astype(np.float32) / 255.0
            rgb = cv2.cvtColor(image[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        else:
            alpha = np.ones(image.shape[:2], np.float32)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        model = alpha > 0.5
        if not model.any():
            continue
        lum = luminance(rgb)[model]
        per_view[view] = {
            "model_pixels": int(model.sum()),
            "mean": round(float(lum.mean()), 4),
            "median": round(float(np.median(lum)), 4),
            "p10": round(float(np.percentile(lum, 10)), 4),
            "p90": round(float(np.percentile(lum, 90)), 4),
            "pct_below_0.05": round(float((lum < 0.05).mean() * 100), 2),
            "pct_below_0.10": round(float((lum < 0.10).mean() * 100), 2),
        }
    report["modes"][mode] = per_view

# observed vs synthesized luminance measured directly on the atlas
atlas = cv2.imread(ATLAS, cv2.IMREAD_UNCHANGED)
cov = cv2.imread(COVERAGE, cv2.IMREAD_GRAYSCALE)
if atlas is not None and cov is not None:
    if cov.shape[:2] != atlas.shape[:2]:
        cov = cv2.resize(cov, (atlas.shape[1], atlas.shape[0]), interpolation=cv2.INTER_NEAREST)
    rgb = cv2.cvtColor(atlas[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    lum = luminance(rgb)
    observed = cov == 255
    synthesized = cov == 130
    report["atlas"] = {
        "observed": {
            "pixels": int(observed.sum()),
            "mean": round(float(lum[observed].mean()), 4) if observed.any() else None,
            "median": round(float(np.median(lum[observed])), 4) if observed.any() else None,
            "pct_below_0.05": round(float((lum[observed] < 0.05).mean() * 100), 2) if observed.any() else None,
        },
        "synthesized": {
            "pixels": int(synthesized.sum()),
            "mean": round(float(lum[synthesized].mean()), 4) if synthesized.any() else None,
            "median": round(float(np.median(lum[synthesized])), 4) if synthesized.any() else None,
            "pct_below_0.05": round(float((lum[synthesized] < 0.05).mean() * 100), 2) if synthesized.any() else None,
        },
    }

emission = report["modes"].get("BASECOLOR_EMISSION", {})
clay = report["modes"].get("NEUTRAL_CLAY", {})
front_em = emission.get("front", {}).get("median")
rear_em = [emission[v]["median"] for v in ("back", "left", "right") if v in emission]
front_clay = clay.get("front", {}).get("median")
rear_clay = [clay[v]["median"] for v in ("back", "left", "right") if v in clay]

verdict = "INCONCLUSIVE"
if front_em and rear_em and front_clay and rear_clay:
    em_ratio = float(np.median(rear_em)) / max(front_em, 1e-6)
    clay_ratio = float(np.median(rear_clay)) / max(front_clay, 1e-6)
    report["front_vs_rear"] = {
        "emission_rear_over_front": round(em_ratio, 4),
        "clay_rear_over_front": round(clay_ratio, 4),
        "note": "emission is lighting-independent, so a low emission ratio is texture darkness; "
                "a low clay ratio with a healthy emission ratio is lighting darkness",
    }
    texture_dark = em_ratio < 0.75
    lighting_dark = clay_ratio < 0.75
    if texture_dark and lighting_dark:
        verdict = "BOTH_TEXTURE_AND_LIGHTING_DARKNESS_CONFIRMED"
    elif texture_dark:
        verdict = "TEXTURE_FILL_DARKNESS_CONFIRMED"
    elif lighting_dark:
        verdict = "LIGHTING_OR_MATERIAL_DARKNESS_CONFIRMED"
    else:
        verdict = "NO_SIGNIFICANT_DARKNESS_ASYMMETRY"
report["verdict"] = verdict

with open(OUT, "w", encoding="utf-8") as handle:
    json.dump(report, handle, indent=2)
print(json.dumps({"verdict": verdict, **report.get("front_vs_rear", {})}, indent=2))
for mode in MODES:
    row = report["modes"].get(mode, {})
    line = " ".join(f"{v}:{row[v]['median']:.3f}" for v in VIEWS if v in row)
    print(f"{mode:20s} median-luminance {line}")
if "atlas" in report:
    print("atlas observed median", report["atlas"]["observed"]["median"],
          "| synthesized median", report["atlas"]["synthesized"]["median"])
