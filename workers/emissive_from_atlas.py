"""Promote small warm bright clusters in a base-colour atlas to emission.

Native texturing bakes lit windows as *albedo*: the right colour lands in the
right place, but a window reads as a beige patch rather than a light source,
because nothing in the material says it emits. A vision review of the gothic
castle put it exactly this way -- "a few warm window-like pixels survived, but
not as convincing emissive light; they read mostly as coloured albedo".

The fix is generic and lives entirely in colour statistics. Nothing here knows
what a window is, or what the subject is:

    a pixel is emissive if it is bright, warm, and part of a SMALL cluster

The size constraint is what makes it safe. Warm and bright alone also describes
sunlit sandstone, a copper roof, a wooden hull -- large connected regions. A lit
window, a lantern, a porthole, a rune, a screen, a cockpit light is a *small*
island of warm brightness surrounded by darker, cooler material. So the filter
is: threshold, label connected components, and keep only components whose area
falls inside a band. Large regions are rejected by the same rule that accepts
windows, without a per-subject exception.

Three guards, because a threshold that fires on everything is worse than one
that fires on nothing:

- **A total-coverage ceiling.** If the selection exceeds a few percent of the
  painted atlas, the thresholds have caught a facade rather than its windows,
  and the run refuses rather than shipping a glowing building.
- **A local-contrast test.** A candidate must be meaningfully brighter than the
  ring of texels around it. This is what separates a window from a bright patch
  of an already-bright wall, and it is scale-free.
- **Emission is masked by the accepted components only**, then feathered. The
  base colour is never modified, so the operation is reversible by deleting the
  emissive texture.

    py emissive_from_atlas.py --mesh asset.glb --out lit.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Luminance floor, on the 0-1 sRGB atlas. Below this a texel is not a light
#: source no matter how warm it is.
MIN_LUMA = 0.42

#: Warmth is (R - B) in sRGB. Deliberately not hue: hue is unstable at low
#: saturation and a dim window sits near grey, where hue is mostly noise.
MIN_WARMTH = 0.055

#: Component area bounds as a fraction of the atlas. The lower bound rejects
#: compression speckle and single stray texels; the upper bound is what keeps
#: this from selecting a sunlit wall. At 1024 these are ~8 and ~2,600 texels.
MIN_COMPONENT_FRACTION = 8e-6
MAX_COMPONENT_FRACTION = 2.5e-3

#: A component must exceed its surrounding ring by this much luminance to count.
#: Without it, every bright patch of a bright material qualifies.
MIN_LOCAL_CONTRAST = 0.10

#: Ring width, in texels, used to measure that surrounding luminance.
CONTRAST_RING = 6

#: If accepted components cover more than this fraction of the *painted* atlas,
#: the thresholds have failed and the tool refuses. Windows are a small part of
#: any facade; if the answer says otherwise, the answer is wrong.
MAX_TOTAL_COVERAGE = 0.045

#: Emission strength and how far the glow is allowed to bleed past the source
#: texels. Feathering is cosmetic -- it stops a hard-edged rectangle of light --
#: and is kept small so it cannot leak across a UV chart boundary.
DEFAULT_GAIN = 1.0
FEATHER_TEXELS = 2


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--gain", type=float, default=DEFAULT_GAIN,
                        help="Emissive factor applied to the accepted texels.")
    parser.add_argument("--min-luma", type=float, default=MIN_LUMA)
    parser.add_argument("--min-warmth", type=float, default=MIN_WARMTH)
    parser.add_argument("--min-contrast", type=float, default=MIN_LOCAL_CONTRAST)
    parser.add_argument("--max-coverage", type=float, default=MAX_TOTAL_COVERAGE)
    parser.add_argument("--dump-mask", default="",
                        help="Write the accepted mask as a PNG, to check what "
                             "the thresholds actually selected.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh
    from PIL import Image
    from scipy import ndimage

    scene = trimesh.load(args.mesh, process=False)
    mesh = scene.to_geometry() if hasattr(scene, "geometry") else scene
    material = getattr(mesh.visual, "material", None)
    texture = getattr(material, "baseColorTexture", None)
    if texture is None:
        raise SystemExit("NO_BASE_COLOUR_TEXTURE: nothing to analyse")

    rgba = np.asarray(texture.convert("RGBA"), dtype=np.float32) / 255.0
    rgb, alpha = rgba[..., :3], rgba[..., 3]
    painted = alpha > 0.5
    height, width = rgb.shape[:2]
    area = float(height * width)

    luma = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    warmth = rgb[..., 0] - rgb[..., 2]

    seed = painted & (luma > args.min_luma) & (warmth > args.min_warmth)

    # Local contrast: compare each candidate against a ring of surrounding
    # texels. A uniform-mean blur is enough -- the question is only "is this
    # brighter than its neighbourhood", and the ring is wide relative to a
    # window so the window's own texels do not dominate its own reference.
    background = ndimage.uniform_filter(
        np.where(painted, luma, 0.0), size=2 * CONTRAST_RING + 1)
    weight = ndimage.uniform_filter(
        painted.astype(np.float32), size=2 * CONTRAST_RING + 1)
    background = background / np.clip(weight, 1e-6, None)
    seed &= (luma - background) > args.min_contrast

    labels, count = ndimage.label(seed)
    accepted = np.zeros_like(seed)
    kept = 0
    if count:
        sizes = np.bincount(labels.reshape(-1))
        sizes[0] = 0
        fraction = sizes / area
        keep = ((fraction >= MIN_COMPONENT_FRACTION)
                & (fraction <= MAX_COMPONENT_FRACTION))
        accepted = keep[labels]
        kept = int(keep.sum())

    painted_texels = int(painted.sum())
    coverage = float(accepted.sum() / max(painted_texels, 1))

    receipt = {
        "schema_version": "emissive_from_atlas_v1",
        "mesh": str(Path(args.mesh).resolve()),
        "atlas_size": [width, height],
        "painted_texels": painted_texels,
        "candidate_texels": int(seed.sum()),
        "components_found": int(count),
        "components_kept": kept,
        "accepted_texels": int(accepted.sum()),
        "coverage_of_painted": round(coverage, 5),
        "thresholds": {
            "min_luma": args.min_luma, "min_warmth": args.min_warmth,
            "min_local_contrast": args.min_contrast,
            "component_fraction": [MIN_COMPONENT_FRACTION,
                                   MAX_COMPONENT_FRACTION],
            "max_total_coverage": args.max_coverage,
        },
        "gain": args.gain,
    }

    if args.dump_mask:
        Image.fromarray((accepted * 255).astype(np.uint8)).save(args.dump_mask)

    if coverage > args.max_coverage:
        receipt["result"] = "refused"
        print(json.dumps(receipt, indent=2))
        raise SystemExit(
            f"EMISSIVE_COVERAGE_TOO_HIGH {coverage:.3%} of painted texels "
            f"exceeds {args.max_coverage:.1%}. The thresholds have selected a "
            f"material, not its light sources. Raise --min-luma or "
            f"--min-contrast, or accept that this subject has none.")
    if kept == 0:
        receipt["result"] = "no_emitters_found"
        print(json.dumps(receipt, indent=2))
        raise SystemExit("NO_EMITTERS_FOUND: no warm bright small clusters "
                         "passed. This is a normal outcome for a subject with "
                         "no lights; nothing was written.")

    # Emission takes the base colour of the accepted texels, so a warm window
    # glows warm and a cool one glows cool. Inventing a colour here would be
    # the subject-specific rule this is meant to avoid.
    glow = np.zeros_like(rgb)
    glow[accepted] = rgb[accepted]
    if FEATHER_TEXELS > 0:
        soft = ndimage.gaussian_filter(glow, sigma=(FEATHER_TEXELS,
                                                    FEATHER_TEXELS, 0))
        # Keep the core at full strength and let only the surround soften, so
        # the window itself does not lose its edge.
        glow = np.maximum(glow, soft * accepted[..., None].astype(np.float32))
        glow = np.maximum(glow, soft)
    glow = np.clip(glow * args.gain, 0.0, 1.0)

    emissive = Image.fromarray((glow * 255).astype(np.uint8), mode="RGB")
    material.emissiveTexture = emissive
    material.emissiveFactor = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(str(out))

    receipt["result"] = "written"
    receipt["output"] = str(out.resolve())
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
