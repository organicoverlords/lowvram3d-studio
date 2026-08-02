"""Render compact PNG proof images from saved MoGe NumPy maps."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from lowvram3d.image_world.moge_preview import save_moge_previews


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    import cv2

    summary = save_moge_previews(
        Path(args.geometry),
        Path(args.output),
        cv2_module=cv2,
    )
    print(summary.to_json(), end="")
    return 0


if __name__ == "__main__":
    sys.exit(main())
