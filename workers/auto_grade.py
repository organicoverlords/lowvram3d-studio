"""Derive grade parameters by matching the atlas to the source image.

The grading values used so far came from a vision model. That worked -- three
times out of three the model's numbers beat a local guess -- but it makes a
required pipeline stage depend on an external call and a human reading the
answer. Both source image and generated atlas are pixel distributions, and the
grade is just the transform that maps one onto the other, so it can be measured.

Everything is computed on the *subject* only: source pixels inside the matte,
atlas texels that are actually painted. Comparing a subject against an atlas
that includes empty gutter would bias every statistic toward the gutter's
colour.

Five parameters, each from one comparison:

    warmth      difference in (R-B) after equalising luminance
    saturation  ratio of mean chroma
    contrast    ratio of the p05-p95 luminance spread
    shadows     where the atlas p05 sits relative to the source p05
    highlights  where the atlas p95 sits relative to the source p95

Two things make this honest rather than merely automatic:

**The source is a photograph or painting with its own lighting; the atlas is
albedo.** A source carries baked shadow that an albedo map should not have, so
matching luminance spread exactly would bake the source's lighting into the
texture. Contrast is therefore matched only partially, via `CONTRAST_TRANSFER`.

**Every output is clamped.** An atlas that failed badly can differ from the
source by any amount, and an unclamped match would return a grade that destroys
it. The clamps are the same ranges the vision-model grades fell in.

Validation: run `--check` against a known grade to see how close this lands.

    py auto_grade.py --source matte.png --mesh asset.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: How much of the source's luminance spread to transfer. The source has baked
#: lighting the albedo should not inherit, so this is deliberately partial.
CONTRAST_TRANSFER = 0.55

#: Clamps. These bound every returned value to the range that hand-checked
#: grades have occupied, so a badly-failed atlas cannot produce a destructive
#: correction.
LIMITS = {
    "contrast": (0.0, 0.40),
    "shadows": (-0.15, 0.10),
    "highlights": (-0.05, 0.25),
    "saturation": (0.90, 1.40),
    "warmth": (-0.10, 0.15),
}

#: Below this alpha a texel is gutter, not surface.
PAINTED_ALPHA = 0.5


def stats(rgb):
    """Luminance percentiles, mean chroma and warmth for a set of pixels."""
    import numpy as np

    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    p05, p50, p95 = (float(np.percentile(luma, q)) for q in (5, 50, 95))
    # Chroma as max-min per pixel: cheap, and unlike HSV saturation it does not
    # explode at low luminance where hue is meaningless.
    chroma = float((rgb.max(axis=1) - rgb.min(axis=1)).mean())
    # Warmth measured after equalising luminance, so a darker atlas does not
    # read as cooler purely for being darker.
    scale = 0.5 / max(p50, 1e-4)
    balanced = np.clip(rgb * scale, 0.0, 1.0)
    warmth = float((balanced[:, 0] - balanced[:, 2]).mean())
    return {"p05": p05, "p50": p50, "p95": p95, "spread": p95 - p05,
            "chroma": chroma, "warmth": warmth}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True,
                        help="Matted RGBA source image. Only pixels inside the "
                             "matte are measured.")
    parser.add_argument("--mesh", required=True,
                        help="Generated GLB whose base colour atlas is graded.")
    parser.add_argument("--contrast-transfer", type=float,
                        default=CONTRAST_TRANSFER)
    parser.add_argument("--check", default="",
                        help="JSON of a reference grade to compare against, "
                             "e.g. '{\"contrast\":0.22,\"warmth\":0.04}'.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image

    source = Image.open(args.source).convert("RGBA")
    array = np.asarray(source, dtype=np.float32) / 255.0
    inside = array[..., 3] > PAINTED_ALPHA
    if not inside.any():
        raise SystemExit("NO_SUBJECT: source has no matte")
    source_stats = stats(array[..., :3][inside])

    scene = trimesh.load(args.mesh, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    texture = getattr(getattr(mesh.visual, "material", None),
                      "baseColorTexture", None)
    if texture is None:
        raise SystemExit("NO_ATLAS: mesh has no base colour texture")
    atlas = np.asarray(texture.convert("RGBA"), dtype=np.float32) / 255.0
    painted = atlas[..., 3] > PAINTED_ALPHA
    if not painted.any():
        painted = np.ones(atlas.shape[:2], dtype=bool)
    atlas_stats = stats(atlas[..., :3][painted])

    # --- derive -----------------------------------------------------------
    # Contrast: how much narrower the atlas spread is than the source's,
    # transferred only partially because the source carries baked lighting.
    ratio = source_stats["spread"] / max(atlas_stats["spread"], 1e-4)
    contrast = (ratio - 1.0) * args.contrast_transfer

    # Shadows and highlights: signed differences at the tails. A positive
    # `shadows` lifts, so an atlas whose blacks sit above the source's needs a
    # negative value to bring them down.
    shadows = source_stats["p05"] - atlas_stats["p05"]
    highlights = source_stats["p95"] - atlas_stats["p95"]

    saturation = source_stats["chroma"] / max(atlas_stats["chroma"], 1e-4)
    warmth = source_stats["warmth"] - atlas_stats["warmth"]

    derived = {"contrast": contrast, "shadows": shadows,
               "highlights": highlights, "saturation": saturation,
               "warmth": warmth}
    grade = {k: round(float(np.clip(v, *LIMITS[k])), 4)
             for k, v in derived.items()}
    clamped = [k for k, v in derived.items()
               if not (LIMITS[k][0] <= v <= LIMITS[k][1])]

    receipt = {
        "schema_version": "auto_grade_v1",
        "source": str(Path(args.source).resolve()),
        "mesh": str(Path(args.mesh).resolve()),
        "source_stats": {k: round(v, 4) for k, v in source_stats.items()},
        "atlas_stats": {k: round(v, 4) for k, v in atlas_stats.items()},
        "contrast_transfer": args.contrast_transfer,
        "grade": grade,
        "clamped": clamped,
        "command": (
            f"--contrast {grade['contrast']} --shadows {grade['shadows']} "
            f"--highlights {grade['highlights']} "
            f"--saturation {grade['saturation']} --warmth {grade['warmth']}"),
    }

    if args.check:
        reference = json.loads(args.check)
        receipt["check"] = {
            k: {"reference": reference[k], "derived": grade[k],
                "delta": round(grade[k] - reference[k], 4)}
            for k in reference if k in grade
        }

    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
