"""Build a low-frequency atlas for a consumer-isolated surface region.

This is a bounded material-routing repair: it does not alter geometry, UVs, or the
canonical atlas. The supplied surface mask is the UV footprint consumed by a
separate material. Protected texels are never used as donors.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--base-atlas", required=True)
    p.add_argument("--surface-mask", required=True)
    p.add_argument("--protected-mask", required=True)
    p.add_argument("--output-atlas", required=True)
    p.add_argument("--report", required=True)
    p.add_argument("--blur-radius", type=int, default=151)
    args = p.parse_args()

    base = cv2.imread(args.base_atlas, cv2.IMREAD_COLOR)
    surface = cv2.imread(args.surface_mask, cv2.IMREAD_GRAYSCALE) > 0
    protected = cv2.imread(args.protected_mask, cv2.IMREAD_GRAYSCALE) > 0
    if base is None or surface.shape != base.shape[:2] or protected.shape != base.shape[:2]:
        raise SystemExit("base atlas and masks must have identical dimensions")

    radius = max(3, int(args.blur_radius))
    if radius % 2 == 0:
        radius += 1
    valid_donors = surface & ~protected & (base.mean(axis=2) >= 8)
    if int(valid_donors.sum()) < 32:
        valid_donors = (~protected) & (base.mean(axis=2) >= 8)
    if int(valid_donors.sum()) < 32:
        raise SystemExit("not enough non-protected donor texels")

    donor = np.median(base[valid_donors], axis=0).astype(np.uint8)
    source = base.copy()
    # Remove all high-frequency information from the isolated footprint before
    # the blur. This prevents facial marks in unclassified overlapping texels from
    # surviving through the blur kernel.
    source[surface] = donor
    low_frequency = cv2.GaussianBlur(source, (radius, radius), 0)
    safe = base.copy()
    safe[surface] = low_frequency[surface]
    safe[surface & (safe.mean(axis=2) < 8)] = donor

    output = Path(args.output_atlas)
    output.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output), safe)
    report = {
        "schema": "surface_safe_atlas_v1",
        "base_atlas": str(args.base_atlas),
        "surface_mask": str(args.surface_mask),
        "protected_mask": str(args.protected_mask),
        "surface_texels": int(surface.sum()),
        "protected_surface_texels": int((surface & protected).sum()),
        "donor_texels": int(valid_donors.sum()),
        "donor_bgr_median": donor.tolist(),
        "blur_radius": radius,
        "output_atlas": str(output),
        "canonical_atlas_unchanged": True,
        "geometry_uv_unchanged": True,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
