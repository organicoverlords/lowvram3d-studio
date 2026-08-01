"""Stage 6, step 3: pack ORM from the baked maps and a material-class segmentation.

Roughness is not a constant and is not guessed from the base colour alone. The base colour says
what something is made of far better than any single channel does - pale desaturated bone, dark
low-saturation wood, mid-saturation dyed cloth - so material class is inferred from base colour
statistics, then constrained by the geometry the bakes already measured: cavity finds the recessed
grime that reads wetter, AO finds the buried areas that never polish.

Metallic stays at zero everywhere except pixels that are simultaneously bright, desaturated and
sitting on a hard convex edge, because nothing in the source is unambiguously metal except the
staff ring and a few charm fittings. Inventing gold across the ornaments is the standard failure
here and it is worse than having no metal at all.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

# Roughness centres per material class. Everything here is weathered, buried outdoors or both, so
# the whole range sits high. An earlier pass put bone at 0.62 on the theory that bone takes a
# polish; rendered, that turned every pale surface into a broad specular sheen that swallowed the
# grime and stain detail underneath, and a probe render with flat 0.94 roughness read markedly
# better. Localised polish now comes from the cavity term below rather than from a low centre.
CLASS_ROUGHNESS = {
    "cloth": 0.93,
    "bone": 0.80,
    "wood": 0.86,
    "metal": 0.55,
    "organic": 0.85,
}
CLASS_INDEX = {name: i for i, name in enumerate(CLASS_ROUGHNESS)}


def classify(basecolor: np.ndarray, cavity: np.ndarray, ao: np.ndarray) -> np.ndarray:
    """Assign a material class per texel from base-colour statistics."""
    rgb = basecolor.astype(np.float32) / 255.0
    hsv = cv2.cvtColor(basecolor, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, saturation, value = hsv[..., 0] * 2.0, hsv[..., 1] / 255.0, hsv[..., 2] / 255.0
    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32)

    result = np.full(basecolor.shape[:2], CLASS_INDEX["cloth"], np.uint8)
    # Bone: bright and washed out.
    result[(luma > 0.55) & (saturation < 0.30)] = CLASS_INDEX["bone"]
    # Wood: dark, warm and not very saturated - the staff shaft and pole charms.
    result[(luma < 0.34) & (saturation < 0.55) & ((hue < 55) | (hue > 330))] = CLASS_INDEX["wood"]
    # Organic charms: mid luma, warm, more saturated than bone.
    result[(luma >= 0.34) & (luma <= 0.62) & (saturation >= 0.30) & (hue < 60)] = CLASS_INDEX["organic"]
    # Metal: bright, desaturated and on a convex ridge that the cavity bake found.
    ridge = cavity > 0.72
    result[(luma > 0.62) & (saturation < 0.18) & ridge] = CLASS_INDEX["metal"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--ao", required=True)
    parser.add_argument("--cavity", required=True)
    parser.add_argument("--material-id", required=True)
    parser.add_argument("--island", required=True, help="raster_project debug_coverage.png")
    parser.add_argument("--output", required=True)
    parser.add_argument("--class-map", default="")
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    basecolor = cv2.cvtColor(cv2.imread(args.basecolor, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    ao = cv2.imread(args.ao, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    cavity = cv2.imread(args.cavity, cv2.IMREAD_GRAYSCALE).astype(np.float32) / 255.0
    size = basecolor.shape[0]
    for name, array in (("ao", ao), ("cavity", cavity)):
        if array.shape[0] != size:
            raise RuntimeError(f"{name} is {array.shape[0]}px but base colour is {size}px")

    # Everything outside the UV island is gutter, and gutter is black. Left in, it classifies as
    # dark wood and swamps the statistics - it read as 73% wood on the first pass - while telling
    # you nothing, because no texel outside an island is ever sampled.
    island = cv2.imread(args.island, cv2.IMREAD_GRAYSCALE) >= 40
    if island.shape[0] != size:
        raise RuntimeError(f"island map is {island.shape[0]}px but base colour is {size}px")

    classes = classify(basecolor, cavity, ao)
    roughness = np.zeros((size, size), np.float32)
    for name, centre in CLASS_ROUGHNESS.items():
        roughness[classes == CLASS_INDEX[name]] = centre

    # Recessed, grimy areas read damp and lose roughness; exposed ridges wear smooth a little.
    roughness -= (1.0 - cavity) * 0.08
    roughness -= (1.0 - ao) * 0.04
    # Per-texel break-up so nothing reads as a flat plastic constant.
    grain = cv2.GaussianBlur(
        np.random.default_rng(12345).normal(0.0, 1.0, (size, size)).astype(np.float32), (0, 0), 2.0
    )
    grain /= max(float(np.abs(grain).max()), 1e-6)
    roughness += grain * 0.05
    # Floor at 0.45: below that this asset develops mirror-like patches that no part of the source
    # supports, and a low roughness floor is what made the first pass look like wet plastic.
    roughness = np.clip(roughness, 0.45, 0.99)

    metallic = np.where(classes == CLASS_INDEX["metal"], 0.60, 0.0).astype(np.float32)
    metallic = cv2.GaussianBlur(metallic, (0, 0), 1.2)
    metallic[metallic < 0.10] = 0.0

    orm = np.dstack([
        np.clip(ao * 255.0, 0, 255),
        np.clip(roughness * 255.0, 0, 255),
        np.clip(metallic * 255.0, 0, 255),
    ]).astype(np.uint8)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, cv2.cvtColor(orm, cv2.COLOR_RGB2BGR))

    if args.class_map:
        palette = np.array([
            [70, 110, 150],   # cloth
            [225, 220, 200],  # bone
            [110, 80, 50],    # wood
            [200, 190, 120],  # metal
            [140, 120, 90],   # organic
        ], np.uint8)
        cv2.imwrite(args.class_map, cv2.cvtColor(palette[classes], cv2.COLOR_RGB2BGR))

    total = float(island.sum())
    report = {
        "output": args.output,
        "resolution": size,
        "packing": {"R": "ao", "G": "roughness", "B": "metallic"},
        "island_pixel_percent": round(float(island.mean() * 100), 3),
        "statistics_scope": "uv_island_only",
        "class_pixel_percent": {
            name: round(float(((classes == index) & island).sum()) / total * 100, 3)
            for name, index in CLASS_INDEX.items()
        },
        "roughness": {
            "min": round(float(roughness[island].min()), 4),
            "max": round(float(roughness[island].max()), 4),
            "mean": round(float(roughness[island].mean()), 4),
            "std": round(float(roughness[island].std()), 4),
        },
        "metallic": {
            "mean": round(float(metallic[island].mean()), 5),
            "percent_above_0_5": round(float((metallic[island] > 0.5).mean() * 100), 4),
        },
        "ao": {"mean": round(float(ao[island].mean()), 4)},
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"ORM roughness mean={roughness[island].mean():.3f} std={roughness[island].std():.3f} "
        f"metallic>0.5={float((metallic[island] > 0.5).mean()*100):.3f}% classes="
        + ",".join(f"{n}:{report['class_pixel_percent'][n]:.1f}%" for n in CLASS_INDEX),
        flush=True,
    )


if __name__ == "__main__":
    main()
