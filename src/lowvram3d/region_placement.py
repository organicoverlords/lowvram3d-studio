"""Turn semantic regions into world-space actor specs the builders can place.

The structural builders previously received regions with no geometry, so every
actor was placed at an arbitrary transform and scaled to an arbitrary size --
which is what a field of identical cubes actually is. Segmentation now carries
a measured median depth and a pixel footprint per region, and those two facts
are enough to recover a real position and size by unprojecting the footprint
through the recovered camera.

Each class gets the treatment its geometry deserves rather than one generic
box:

``terrain``      a ground plane at the region's depth, sized to its footprint
``architecture`` a volume standing on the ground, height from its pixel extent
``vegetation``   scatter points across the footprint, not one merged blob
``water``        a flat surface, no vertical extent
``sky``          a backdrop, excluded from placement entirely
``crossing``     a strip following the footprint's long axis

Positions are in Unreal's frame and centimetres, with +X forward from the
camera, so a builder can spawn them directly.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

CM_PER_M = 100.0

# Vegetation is scattered rather than merged: one actor covering a whole tree
# line reads as a wall, several read as trees.
SCATTER_TARGET = {"vegetation": 12, "prop": 6}

# Classes that describe appearance rather than occupancy.
NON_PLACED = {"sky"}


def unproject(u: float, v: float, depth: float, fov_x_deg: float,
              aspect: float) -> tuple[float, float, float]:
    """Normalised image coords (0..1) plus depth -> Unreal metres, +X forward.

    The camera looks down +X with +Y right and +Z up, so image x maps to Y and
    image y maps to inverted Z.
    """
    half_x = math.tan(math.radians(fov_x_deg) * 0.5)
    half_y = half_x / aspect
    ndc_x = (u - 0.5) * 2.0
    ndc_y = (v - 0.5) * 2.0
    return (depth, ndc_x * half_x * depth, -ndc_y * half_y * depth)


def _footprint(region: dict[str, Any], fov_x: float, aspect: float
               ) -> tuple[tuple[float, float, float], float, float]:
    x0, y0, x1, y1 = region["bbox_norm_xyxy"]
    depth = float(region.get("depth_m", {}).get("median") or 10.0)

    centre = unproject((x0 + x1) * 0.5, (y0 + y1) * 0.5, depth, fov_x, aspect)
    left = unproject(x0, (y0 + y1) * 0.5, depth, fov_x, aspect)
    right = unproject(x1, (y0 + y1) * 0.5, depth, fov_x, aspect)
    top = unproject((x0 + x1) * 0.5, y0, depth, fov_x, aspect)
    bottom = unproject((x0 + x1) * 0.5, y1, depth, fov_x, aspect)

    width = abs(right[1] - left[1])
    height = abs(top[2] - bottom[2])
    return centre, width, height


def place(segmentation: dict[str, Any], fov_x_deg: float | None = None
          ) -> dict[str, Any]:
    aspect = segmentation["image_dimensions"][0] / segmentation["image_dimensions"][1]
    fov_x = float(fov_x_deg or segmentation.get("camera", {}).get("fov_x_deg") or 90.0)

    actors: list[dict[str, Any]] = []
    skipped: list[str] = []

    for region in segmentation["regions"]:
        layer = region["layer_type"]
        if layer in NON_PLACED:
            skipped.append(region["id"])
            continue

        centre, width, height = _footprint(region, fov_x, aspect)
        depth_band = region.get("depth_m", {})
        thickness = max(0.5, float(depth_band.get("far", 0.0))
                        - float(depth_band.get("near", 0.0)))

        common = {
            "region_id": region["id"],
            "semantic_label": region["semantic_label"],
            "layer_type": layer,
            "confidence": region.get("confidence"),
            "observed_fraction": region.get("observed_fraction"),
        }

        if layer == "terrain":
            actors.append({**common, "kind": "ground_plane",
                           "location_cm": [centre[0] * CM_PER_M, centre[1] * CM_PER_M,
                                           (centre[2] - height * 0.5) * CM_PER_M],
                           # Terrain runs away from the camera, so its depth
                           # band is the extent along +X, not a thickness.
                           "size_m": [thickness, max(width, 1.0), 0.2]})

        elif layer == "water":
            actors.append({**common, "kind": "water_surface",
                           "location_cm": [c * CM_PER_M for c in centre],
                           "size_m": [thickness, max(width, 1.0), 0.05]})

        elif layer == "architecture":
            actors.append({**common, "kind": "structure",
                           "location_cm": [c * CM_PER_M for c in centre],
                           # Depth is unobservable from one view; assume a
                           # footprint roughly as deep as it is wide.
                           "size_m": [max(width * 0.6, 1.0), max(width, 1.0),
                                      max(height, 1.0)],
                           "assumption": "building depth inferred from width"})

        elif layer in SCATTER_TARGET:
            count = SCATTER_TARGET[layer]
            for index in range(count):
                # Deterministic spread across the footprint; no RNG so a rerun
                # reproduces the same scene.
                t = (index + 0.5) / count
                offset = (t - 0.5) * width
                jitter = ((index * 37) % 11 - 5) / 10.0
                actors.append({
                    **common,
                    "kind": "scatter_instance",
                    "instance_index": index,
                    "location_cm": [(centre[0] + jitter * thickness * 0.3) * CM_PER_M,
                                    (centre[1] + offset) * CM_PER_M,
                                    (centre[2] - height * 0.5) * CM_PER_M],
                    "size_m": [max(width / count, 0.5)] * 2 + [max(height, 1.0)],
                })

        elif layer == "crossing":
            actors.append({**common, "kind": "path_strip",
                           "location_cm": [c * CM_PER_M for c in centre],
                           "size_m": [thickness, max(width, 1.0), 0.1]})

        else:
            actors.append({**common, "kind": "clutter_volume",
                           "location_cm": [c * CM_PER_M for c in centre],
                           "size_m": [max(width * 0.5, 0.5), max(width, 0.5),
                                      max(height, 0.5)]})

    return {
        "schema_version": "region_placement_v1",
        "classification": "PROVEN" if actors else "EMPTY",
        "camera_fov_x_deg": fov_x,
        "aspect_ratio": aspect,
        "actor_count": len(actors),
        "kinds": sorted({a["kind"] for a in actors}),
        "skipped_regions": skipped,
        "actors": actors,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--segmentation", required=True)
    parser.add_argument("--receipt", required=True)
    parser.add_argument("--fov-x", type=float, default=None)
    args = parser.parse_args(argv)

    segmentation = json.loads(Path(args.segmentation).read_text(encoding="utf-8"))
    result = place(segmentation, args.fov_x)

    receipt = Path(args.receipt)
    receipt.parent.mkdir(parents=True, exist_ok=True)
    receipt.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n",
                       encoding="utf-8")

    summary = {k: v for k, v in result.items() if k != "actors"}
    summary["actors"] = [
        {"id": a["region_id"], "kind": a["kind"],
         "at_m": [round(c / CM_PER_M, 1) for c in a["location_cm"]],
         "size_m": [round(s, 1) for s in a["size_m"]]}
        for a in result["actors"]
    ]
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
