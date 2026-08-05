"""Crop a candidate render to the same framing as the stored fixture crops.

The deterministic gate compares a candidate against a stored `before` crop. Both must be cut from
the same box, or the comparison measures framing rather than the repair: an uncropped 1024px
render against a FRONT_BOX crop reports ~59% outside-ROI change even for a perfect repair.
"""
from __future__ import annotations

import sys
from pathlib import Path

from PIL import Image

# Must match evidence/visual-qa/build_fixtures.py FRONT_BOX.
FRONT_BOX = (300, 300, 740, 740)
MAX_SIDE = 512


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: crop_for_gate.py <source.png> <target.png>", file=sys.stderr)
        return 2
    source, target = Path(sys.argv[1]), Path(sys.argv[2])
    image = Image.open(source).convert("RGB").crop(FRONT_BOX)
    if max(image.size) > MAX_SIDE:
        scale = MAX_SIDE / float(max(image.size))
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
            Image.LANCZOS,
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    image.save(target)
    print(f"CROPPED {source} -> {target} {image.size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
