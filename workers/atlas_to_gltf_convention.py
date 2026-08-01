"""Convert a raster_project atlas into glTF texture orientation.

raster_project.py rasterises into the atlas with `row = (1 - v) * (size - 1)`, so the texel for
v=0 lands on the LAST row. glTF samples the other way round - v=0 is the FIRST row - and the maps
baked through Blender already follow that convention, because Blender's importer flips V on the way
in and its PNG writer flips rows on the way out, and the two cancel.

Left unconverted the base colour is vertically mirrored against its own UVs, so every chart samples
some unrelated chart's pixels. It does not look like a flip: it looks like a plausible patchwork,
which is why this is a conversion step with a test behind it rather than a comment.

Verified numerically on the shaman: sampling the converted atlas at row = v*(size-1) reproduces the
projected source colour to a mean absolute error of 3.2/255, against 61.9/255 unconverted.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.input, cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError(f"could not read {args.input}")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(args.output, image[::-1])
    print(f"ATLAS_CONVERTED {args.input} -> {args.output} ({image.shape[1]}x{image.shape[0]})", flush=True)


if __name__ == "__main__":
    main()
