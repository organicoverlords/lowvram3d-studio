"""Stage 6 repair: re-inject surface detail into synthesized atlas regions.

raster_project resolves unseen surfaces to one flat colour per triangle and then diffuses across
each chart, which is the right call for colour - it cannot invent detail it never observed - but it
leaves those regions reading as smooth untextured clay next to the observed areas.

The detail added here is not invented: it is the high-frequency component of the cavity and AO
bakes, which were measured from the real high-poly surface. That means crevices, pitting and wear
already present in the geometry become visible in the base colour, at the same places the shading
maps already darken. Nothing semantic is introduced - no eyes, symbols or ornaments - because a
high-pass of a geometry bake carries no such structure.

Observed texels are left untouched. Only the synthesized region is modulated, so the front stays
exactly as projected.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def high_pass(image: np.ndarray, sigma: float, clip: float) -> np.ndarray:
    """Zero-mean high-frequency component, scaled to roughly unit range.

    Clipped rather than left unbounded: the tails of a cavity high-pass are thin crevice lines, and
    at the strength needed to match observed contrast they would otherwise drive the gain negative.
    """
    low = cv2.GaussianBlur(image, (0, 0), sigma)
    detail = image - low
    scale = float(np.percentile(np.abs(detail), 99)) or 1.0
    return np.clip(detail / scale, -clip, clip)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--coverage", required=True, help="glTF-oriented debug_coverage.png")
    parser.add_argument("--cavity", required=True)
    parser.add_argument("--ao", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--strength", type=float, default=0.34)
    parser.add_argument("--sigma", type=float, default=6.0)
    parser.add_argument("--clip", type=float, default=1.0)
    args = parser.parse_args()

    basecolor = cv2.imread(args.basecolor, cv2.IMREAD_COLOR).astype(np.float32) / 255.0
    coverage = cv2.imread(args.coverage, cv2.IMREAD_GRAYSCALE)
    cavity = cv2.imread(args.cavity, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    ao = cv2.imread(args.ao, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    size = basecolor.shape[0]
    for name, array in (("coverage", coverage), ("cavity", cavity), ("ao", ao)):
        if array.shape[0] != size:
            raise RuntimeError(f"{name} is {array.shape[0]}px but base colour is {size}px")

    island = coverage >= 40
    observed = coverage >= 255
    synthesized = island & ~observed

    detail = 0.72 * high_pass(cavity, args.sigma, args.clip) + 0.28 * high_pass(ao, args.sigma, args.clip)
    # Fade the modulation out towards the observed boundary so the join is not a visible step.
    ramp = cv2.GaussianBlur(synthesized.astype(np.float32), (0, 0), 3.0)
    gain = 1.0 + args.strength * detail * ramp

    result = basecolor.copy()
    modulated = basecolor * gain[..., None]
    result[synthesized] = modulated[synthesized]
    result = np.clip(result, 0.0, 1.0)

    cv2.imwrite(args.output, (result * 255.0).astype(np.uint8))

    before = basecolor[synthesized]
    after = result[synthesized]

    def local_contrast(image):
        grey = image.mean(axis=2)
        return float(np.abs(grey - cv2.GaussianBlur(grey, (0, 0), 4.0))[synthesized].std())

    report = {
        "basecolor": args.basecolor,
        "output": args.output,
        "strength": args.strength,
        "sigma": args.sigma,
        "island_pixels": int(island.sum()),
        "observed_pixels": int(observed.sum()),
        "synthesized_pixels": int(synthesized.sum()),
        "synthesized_percent_of_island": round(float(synthesized.sum() / max(island.sum(), 1) * 100), 2),
        "observed_unchanged": bool(np.array_equal(basecolor[observed], result[observed])),
        "synthesized_mean_before": [round(float(v), 4) for v in before.mean(axis=0)],
        "synthesized_mean_after": [round(float(v), 4) for v in after.mean(axis=0)],
        "local_contrast_before": round(local_contrast(basecolor), 5),
        "local_contrast_after": round(local_contrast(result), 5),
        "local_contrast_observed_target": round(
            float(np.abs(basecolor.mean(axis=2) - cv2.GaussianBlur(basecolor.mean(axis=2), (0, 0), 4.0))[observed].std()),
            5,
        ),
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"DETAIL_FILL synthesized={report['synthesized_percent_of_island']}% "
        f"contrast {report['local_contrast_before']} -> {report['local_contrast_after']} "
        f"observed_unchanged={report['observed_unchanged']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
