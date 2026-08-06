"""Square a matted RGBA image for the generator without dissolving thin features.

Downsampling is where thin structures die, and it happens before the model ever
sees them. A hanging cord one pixel wide in a 3072 px source is one sixth of a
pixel at 512; area-weighted resampling spreads its coverage over the whole
destination texel and returns alpha near 0.15. The generator then sees a faint
smudge and builds nothing. The cord was never lost by the model -- it was lost
by PIL.

So alpha is resampled with a **max-biased** pool rather than an average: a
destination texel is opaque if any meaningful part of its source footprint was
opaque. Colour is still resampled normally, because averaging colour is correct
and only coverage is at issue.

That trade is deliberate and it is not free. Max-pooling alpha thickens every
silhouette by up to half a destination texel, which on a large smooth subject
reads as a slightly bloated outline. `--thin-preserve` is therefore a blend
between the averaged and maxed alpha, not a hard switch, and it defaults to a
value that rescues one-pixel features without visibly fattening a hull.

Framing matters for a second reason, measured on this stack. The generator's
sparse-structure stage runs on a 32^3 grid at `--res 512`, so one structural
cell is `512 / 32 = 16` source pixels. A feature smaller than that competes for
a single cell and usually loses:

    hanging strings   1 px wide but ~150 px long   ~10 cells along the run   survived
    pendants          15-23 px                     ~1-1.5 cells             blobbed
    portcullis bars   ~3 px                        0.2 cells                melted

Length is what saves the strings: a long thin feature claims cells along its
run even at sub-cell width, while a small compact prop cannot claim even one.
This tool reports the structural-cell size for the framing it produced, so that
prediction can be made before spending four minutes on a run.

    py prepare_input.py --image matte.png --out crop512.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Fraction of the subject's largest extent added as margin on the square
#: canvas. The generator extends geometry to meet a frame edge, so the subject
#: must not touch it; beyond that, every pixel of margin is resolution thrown
#: away, which on a threshold-limited subject is exactly what you cannot spare.
DEFAULT_MARGIN = 0.04

#: Blend between area-averaged alpha (0.0) and max-pooled alpha (1.0).
DEFAULT_THIN_PRESERVE = 0.65

#: An enclosed hole must be at least this fraction of a structural cell across
#: before it is treated as a designed opening worth widening. Below it, a hole
#: is matte speckle, and widening speckle perforates the subject.
MIN_HOLE_CELLS = 0.25

#: Structural grid the sparse-structure stage runs on.
#:
#: **32 at every --res.** This was assumed to scale with --res and it does not:
#: a res-1024 run logs `active voxels @res32` exactly like a res-512 run, then
#: does `LR 512 -> upsample -> HR 1024 cascade`. Occupancy is decided once, on a
#: 32^3 grid, and everything above that refines what the grid already claimed.
#:
#: The consequence is the useful part: **raising --res cannot recover a feature
#: that failed to claim a structural cell.** It can only sharpen one that did.
#: A pendant that came out as a stub had a cell and may be improved; a
#: portcullis bar that vanished had none and will stay vanished.
STRUCTURAL_GRID = 32


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True, help="Matted RGBA source.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN)
    parser.add_argument("--thin-preserve", type=float,
                        default=DEFAULT_THIN_PRESERVE,
                        help="0 = plain resampling, 1 = pure max-pooled alpha.")
    parser.add_argument("--res", type=int, default=512, choices=(512, 1024, 1536),
                        help="Generator --res this input is destined for. "
                             "Recorded only; it does NOT change the structural "
                             "cell size, which is fixed by the 32^3 occupancy "
                             "grid and the input resolution.")
    parser.add_argument("--emphasise", type=float, default=0.0,
                        help="NEGATIVE RESULT, left in place because the reason "
                             "it fails is worth keeping. Widens enclosed holes "
                             "smaller than a structural cell, on the theory "
                             "that sub-cell openings fill in during generation "
                             "and widening them would preserve a pendant's "
                             "shape without a detail pass. It does not work: "
                             "widening a 7 px hole to 16 px removes ~9 px of "
                             "material all round, and the pendant wall is 8 px "
                             "thick, so the hole eats through the prop. A "
                             "feature whose whole body spans ~2 cells cannot "
                             "contain a 1-cell hole. Measured at 1.0 on the "
                             "shaman: the ovoid's hole became a gash and the "
                             "hook pendant disappeared. No setting helps; the "
                             "arithmetic is the problem.")
    parser.add_argument("--emphasise-positive", type=float, default=0.0,
                        help="Thicken thin SOLID features too. Off by default "
                             "and rarely wanted: adjacent thin features merge, "
                             "which is how a robe fringe becomes a sheet.")
    parser.add_argument("--crop", default="",
                        help="x0,y0,x1,y1 in NORMALISED coordinates of the "
                             "full-frame square input, for a detail pass. The "
                             "crop is taken from the full-resolution source "
                             "before downsampling, so the feature really does "
                             "gain structural cells rather than being "
                             "upsampled from an already-lossy 512.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    from PIL import Image

    image = Image.open(args.image).convert("RGBA")
    alpha_full = np.asarray(image)[..., 3]
    if not (alpha_full > 8).any():
        raise SystemExit("NO_ALPHA: input is opaque; matte it first")

    ys, xs = np.nonzero(alpha_full > 8)
    crop = image.crop((int(xs.min()), int(ys.min()),
                       int(xs.max()) + 1, int(ys.max()) + 1))
    width, height = crop.size
    side = int(max(width, height) * (1.0 + 2 * args.margin))
    canvas = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    canvas.paste(crop, ((side - width) // 2, (side - height) // 2), crop)

    # Detail pass. Cut the region out of the full-resolution square canvas, so
    # what reaches the generator is genuinely more pixels on the feature rather
    # than an upsample of the already-downsampled full-frame input.
    detail = None
    if args.crop:
        try:
            fractions = [float(v) for v in args.crop.split(",")]
            if len(fractions) != 4:
                raise ValueError
        except ValueError:
            raise SystemExit("BAD_CROP: expected x0,y0,x1,y1 as fractions")
        cx0, cy0, cx1, cy1 = (int(round(f * side)) for f in fractions)
        cx0, cy0 = max(cx0, 0), max(cy0, 0)
        cx1, cy1 = min(cx1, side), min(cy1, side)
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            raise SystemExit("BAD_CROP: region is empty")
        region = canvas.crop((cx0, cy0, cx1, cy1))
        # Re-square the region on its own bounding box, so the feature fills the
        # frame rather than sitting in a letterboxed strip -- a strip would
        # magnify nothing, which is the trap the first crop suggestion fell into.
        region_alpha = np.asarray(region)[..., 3]
        if (region_alpha > 8).any():
            rys, rxs = np.nonzero(region_alpha > 8)
            region = region.crop((int(rxs.min()), int(rys.min()),
                                  int(rxs.max()) + 1, int(rys.max()) + 1))
        rw, rh = region.size
        rside = int(max(rw, rh) * (1.0 + 2 * args.margin))
        squared = Image.new("RGBA", (rside, rside), (0, 0, 0, 0))
        squared.paste(region, ((rside - rw) // 2, (rside - rh) // 2), region)
        detail = {"requested_px": [cx0, cy0, cx1, cy1],
                  "region_px": [rw, rh],
                  "magnification": round(side / max(rside, 1), 2)}
        canvas, side = squared, rside

    size = args.size
    averaged = canvas.resize((size, size), Image.LANCZOS)

    # Max-pooled alpha at the same destination grid. Done by reducing with
    # NEAREST after a local maximum filter, so a single opaque source pixel
    # anywhere in a destination footprint survives.
    from scipy import ndimage
    factor = side / size
    reach = max(int(round(factor / 2)), 1)
    alpha_source = np.asarray(canvas)[..., 3]
    maxed = ndimage.maximum_filter(alpha_source, size=2 * reach + 1)
    maxed = np.asarray(
        Image.fromarray(maxed).resize((size, size), Image.NEAREST),
        dtype=np.float32)

    out = np.asarray(averaged, dtype=np.float32).copy()
    blend = float(np.clip(args.thin_preserve, 0.0, 1.0))
    out[..., 3] = (1.0 - blend) * out[..., 3] + blend * maxed

    # Structure-aware pre-emphasis, applied at the OUTPUT resolution because
    # that is where the structural cell is defined.
    emphasis = None
    if args.emphasise > 0 or args.emphasise_positive > 0:
        cell = size / STRUCTURAL_GRID
        radius = max(int(round(cell / 2.0)), 1)
        yy, xx = np.mgrid[-radius:radius + 1, -radius:radius + 1]
        disc = (xx ** 2 + yy ** 2) <= radius ** 2
        solid = out[..., 3] > 128

        widened = 0
        if args.emphasise > 0:
            # ENCLOSED holes only.
            #
            # The obvious rule -- `closing & ~solid` -- was tried and it
            # destroys the subject. It captures every silhouette concavity: the
            # gaps between fringe strips, between the legs, around the robe.
            # Dilating those carved the beak off and punched holes through the
            # figure, taking coverage from 0.289 to 0.164.
            #
            # A hole is a background component that does not touch the frame
            # border. That separates the hole through a pendant from the gap
            # between two legs, which morphology alone cannot. Only holes are
            # widened, so the silhouette is never touched.
            background = ~solid
            comps, ncomp = ndimage.label(background)
            border = set(np.unique(np.concatenate([
                comps[0, :], comps[-1, :], comps[:, 0], comps[:, -1]])))
            border.discard(0)
            enclosed = np.isin(comps, [c for c in range(1, ncomp + 1)
                                       if c not in border])
            # Only those under a cell across -- larger ones already survive.
            small = np.zeros_like(enclosed)
            hole_labels, nholes = ndimage.label(enclosed)
            for h in range(1, nholes + 1):
                region = hole_labels == h
                across = ndimage.distance_transform_edt(region).max() * 2.0
                # Lower bound as well as upper. Without it, every 1-2 px
                # pinhole in the matte gets dilated by half a cell and the
                # subject comes out perforated -- 15% of it carved away on the
                # first attempt. A hole worth preserving is a designed opening,
                # not alpha speckle.
                if MIN_HOLE_CELLS * cell < across < cell:
                    small |= region
            grow = max(int(round(radius * args.emphasise)), 1)
            gy, gx = np.mgrid[-grow:grow + 1, -grow:grow + 1]
            gdisc = (gx ** 2 + gy ** 2) <= grow ** 2
            opened_up = ndimage.binary_dilation(small, structure=gdisc)
            solid_after = solid & ~opened_up
            widened = int((solid & ~solid_after).sum())
            solid = solid_after

        thickened = 0
        if args.emphasise_positive > 0:
            thin = solid & ~ndimage.binary_opening(solid, structure=disc)
            grow = max(int(round(radius * args.emphasise_positive)), 1)
            gy, gx = np.mgrid[-grow:grow + 1, -grow:grow + 1]
            gdisc = (gx ** 2 + gy ** 2) <= grow ** 2
            fattened = ndimage.binary_dilation(thin, structure=gdisc)
            thickened = int((fattened & ~solid).sum())
            solid = solid | fattened

        out[..., 3] = np.where(solid, np.maximum(out[..., 3], 255.0), 0.0)
        emphasis = {"negative": args.emphasise, "positive": args.emphasise_positive,
                    "cell_px": round(cell, 2),
                    "texels_carved": widened, "texels_added": thickened}

    result = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    result.save(args.out)

    final_alpha = np.asarray(result)[..., 3]
    grid = STRUCTURAL_GRID
    cell_px = size / grid

    receipt = {
        "schema_version": "prepare_input_v1",
        "source": str(Path(args.image).resolve()),
        "output": str(Path(args.out).resolve()),
        "source_subject_px": [int(width), int(height)],
        "source_aspect": round(width / height, 3),
        "square_side": side,
        "margin": args.margin,
        "thin_preserve": blend,
        "downsample_factor": round(factor, 2),
        "subject_px_in_output": [round(size * width / side),
                                 round(size * height / side)],
        "coverage": round(float((final_alpha > 8).mean()), 4),
        "coverage_averaged_only": round(
            float((np.asarray(averaged)[..., 3] > 8).mean()), 4),
        "target_res": args.res,
        "structural_grid": grid,
        "structural_cell_px": round(cell_px, 2),
        "detail_pass": detail,
        "emphasis": emphasis,
        "note": (f"A feature smaller than ~{cell_px:.0f} px competes for a "
                 f"single structural cell and usually loses, unless it is long "
                 f"enough to claim cells along its length."),
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
