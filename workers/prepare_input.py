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
