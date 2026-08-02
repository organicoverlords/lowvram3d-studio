"""Create a deterministic synthetic surface-projection fixture for Blender CI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--size", type=int, default=65)
    args = parser.parse_args()
    if args.size < 9:
        raise ValueError("size must be at least nine")

    root = Path(args.output).resolve()
    arrays = root / "arrays"
    arrays.mkdir(parents=True, exist_ok=True)

    axis = np.linspace(-1.0, 1.0, args.size, dtype=np.float64)
    x, y = np.meshgrid(axis, axis)
    island = np.exp(-3.6 * (x * x + y * y))
    ridge = 0.45 * np.exp(-18.0 * ((x + 0.35) ** 2 + (y - 0.10) ** 2))
    channels = 0.12 * np.sin(7.0 * x) * np.cos(5.0 * y)
    height = np.maximum(0.0, island + ridge + channels).astype(np.float32)

    observed = np.zeros((args.size, args.size), dtype=bool)
    observed[:, : args.size // 2 + 5] = True
    observed &= (x * x + y * y) < 1.65
    generated = ~observed
    confidence = np.where(observed, 0.82, 0.0).astype(np.float32)
    confidence[(observed) & (x > 0.15)] = 0.10

    np.save(arrays / "completed-height.npy", height, allow_pickle=False)
    np.save(arrays / "observed-mask.npy", observed.astype(np.uint8), allow_pickle=False)
    np.save(arrays / "generated-mask.npy", generated.astype(np.uint8), allow_pickle=False)
    np.save(arrays / "confidence.npy", confidence, allow_pickle=False)
    (root / "surface-projection-report.json").write_text(
        json.dumps(
            {
                "status": "SYNTHETIC_FIXTURE",
                "classification": "TEST_ONLY_NOT_REAL_TERRAIN",
                "promotion_allowed": False,
                "grid_size": args.size,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
