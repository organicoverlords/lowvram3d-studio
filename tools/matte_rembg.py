"""Cut a subject out of its background with rembg/u2net.

    py tools/matte_rembg.py --image source.jpg --out matte.png --receipt m.json

Must run under the HY3D2 standalone interpreter, which is where rembg and the
u2net weights already live:

    C:/AI/HY3D2/python_standalone/python.exe

This is not a replacement for `workers/auto_matte.py`, it is the other half of
the problem. auto_matte measures texture energy to find the tolerance at which a
colour key stops eating shadow and starts eating the subject, which is the right
instrument for a studio plate on a flat backdrop. It has no answer for a
background that is not flat: on the seal diver's foggy teal gradient it selected
tolerance 0 and kept 73% of the frame, corners included, and that image fed to a
generator produces a slab.

Segmentation does not care whether the background is flat. Use auto_matte when
there is a contact shadow to reason about, and this when there is not.

Two things are checked rather than assumed, because a bad matte is only visible
three stages downstream:

  corners   all four must be transparent, or the background survived
  coverage  a plausible subject share; ~1.0 means nothing was removed

Prints CUDA provider errors to stderr and falls back to CPU. That is expected on
this machine -- onnxruntime wants cudnn64_9.dll, which is not installed -- and
costs a few seconds on a 2048 px image, so it is not worth chasing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="u2net")
    parser.add_argument("--max-side", type=int, default=2048,
                        help="downscale the long side before matting; an 8K "
                             "plate costs minutes and gains no edge accuracy "
                             "the generator can use at 1024 conditioning")
    parser.add_argument("--receipt")
    parser.add_argument("--preview", action="store_true",
                        help="also write <out>_check.png over magenta")
    args = parser.parse_args(argv)

    from rembg import new_session, remove

    image = Image.open(args.image).convert("RGB")
    original = image.size
    if max(image.size) > args.max_side:
        scale = args.max_side / max(image.size)
        image = image.resize((max(1, int(image.width * scale)),
                             max(1, int(image.height * scale))), Image.LANCZOS)

    cut = remove(image, session=new_session(args.model),
                 alpha_matting=True,
                 alpha_matting_foreground_threshold=240,
                 alpha_matting_background_threshold=15,
                 alpha_matting_erode_size=8)

    target = Path(args.out)
    target.parent.mkdir(parents=True, exist_ok=True)
    cut.save(target)

    alpha = np.array(cut)[..., 3]
    opaque = alpha > 128
    coverage = float(opaque.mean())
    corners = [int(alpha[0, 0]), int(alpha[0, -1]),
               int(alpha[-1, 0]), int(alpha[-1, -1])]
    rows, columns = np.where(opaque)
    box = ([int(columns.min()), int(columns.max()),
            int(rows.min()), int(rows.max())] if len(rows) else [0, 0, 0, 0])

    failures = []
    if max(corners) > 128:
        failures.append(f"corner alpha {corners} -- background survived")
    if coverage > 0.92:
        failures.append(f"coverage {coverage:.3f} -- almost nothing was removed")
    if coverage < 0.01:
        failures.append(f"coverage {coverage:.3f} -- the subject was removed")

    print(f"[matte] {original[0]}x{original[1]} -> {cut.size[0]}x{cut.size[1]}, "
          f"coverage {coverage:.4f}, corners {corners}", flush=True)
    print(f"[matte] subject box x {box[0]}..{box[1]} y {box[2]}..{box[3]}",
          flush=True)

    if args.preview:
        backdrop = Image.new("RGBA", cut.size, (255, 0, 255, 255))
        Image.alpha_composite(backdrop, cut).convert("RGB").save(
            target.with_name(target.stem + "_check.png"))

    if args.receipt:
        Path(args.receipt).write_text(json.dumps({
            "schema": "lowvram3d_matte_rembg_v1",
            "image": str(Path(args.image).resolve()),
            "output": str(target.resolve()),
            "model": args.model,
            "source_size": list(original),
            "matte_size": list(cut.size),
            "coverage": round(coverage, 4),
            "corner_alpha": corners,
            "subject_box": box,
            "failures": failures,
            "ok": not failures,
        }, indent=2), encoding="utf-8")

    if failures:
        for failure in failures:
            print(f"MATTE_ABORT: {failure}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
