"""Build a fail-closed semantic mask package from saved probability arrays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import traceback

import numpy as np

from lowvram3d.image_world.semantic_masks import (
    SEMANTIC_CLASSES,
    build_semantic_mask_set,
    mask_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Directory containing <class>.npy arrays")
    parser.add_argument("--output", required=True)
    parser.add_argument("--valid-mask", help="Optional NumPy validity mask")
    parser.add_argument("--terrain-threshold", type=float, default=0.60)
    parser.add_argument("--exclusion-threshold", type=float, default=0.35)
    parser.add_argument("--minimum-margin", type=float, default=0.15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "semantic-mask-report.json"

    try:
        probabilities = {
            name: np.load(source / f"{name}.npy", allow_pickle=False)
            for name in SEMANTIC_CLASSES
        }
        valid = None
        if args.valid_mask:
            valid = np.load(Path(args.valid_mask).resolve(), allow_pickle=False).astype(bool)

        result = build_semantic_mask_set(
            probabilities,
            valid_mask=valid,
            terrain_threshold=args.terrain_threshold,
            exclusion_threshold=args.exclusion_threshold,
            minimum_margin=args.minimum_margin,
        )

        arrays = output / "arrays"
        arrays.mkdir(parents=True, exist_ok=True)
        np.save(arrays / "terrain-candidate.npy", result.terrain_candidate.astype(np.uint8), allow_pickle=False)
        np.save(arrays / "unresolved.npy", result.unresolved.astype(np.uint8), allow_pickle=False)
        np.save(arrays / "semantic-confidence.npy", result.confidence.astype(np.float32), allow_pickle=False)
        np.save(arrays / "semantic-class-index.npy", result.class_index.astype(np.uint8), allow_pickle=False)

        report = mask_report(result)
        report.update({
            "status": "PASS_CONTRACT",
            "source": str(source),
            "classes": list(SEMANTIC_CLASSES),
            "errors": [],
        })
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("IMAGE_WORLD_SEMANTIC_MASKS_PASS_CONTRACT")
        return 0
    except Exception as exc:
        report = {
            "status": "FAILED",
            "classification": "SEMANTIC_MASK_PACKAGE_FAILED",
            "promotion_allowed": False,
            "errors": [f"{type(exc).__name__}: {exc}", traceback.format_exc()],
        }
        report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(report["errors"][0], file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
