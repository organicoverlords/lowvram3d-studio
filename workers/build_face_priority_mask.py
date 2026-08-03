"""Render the configured protected-region mask and its overlay on the source view."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

import protected_region


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--region-config", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    config = protected_region.load(Path(args.region_config))
    size = int(config["source_image_size"])
    masks = protected_region.build_masks(config, size)
    source = np.asarray(Image.open(args.source_image).convert("RGB"))
    if source.shape[0] != size:
        raise RuntimeError(f"FACE_MASK_SOURCE_SIZE_MISMATCH:{source.shape[0]}:{size}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for name, record in masks.items():
        Image.fromarray((record["weight"] * 255).astype(np.uint8)).save(
            output_dir / f"{name}_priority_mask.png")
        records[name] = {
            "owner_semantic": record["owner_semantic"],
            "forbidden_owner_semantics": record["forbidden_owner_semantics"],
            "priority": record["priority"],
            "hard_pixels": int(record["hard"].sum()),
            "feathered_weight_sum": float(record["weight"].sum()),
            "coverage_fraction": float(record["hard"].mean()),
            "mask": str(output_dir / f"{name}_priority_mask.png"),
        }
    protected_region.overlay(source, masks).save(output_dir / "face_priority_mask_overlay.png")

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps({
        "schema": "protected_region_mask_v1",
        "region_config": str(args.region_config),
        "source_image": str(args.source_image),
        "source_image_size": size,
        "overlay": str(output_dir / "face_priority_mask_overlay.png"),
        "regions": records,
    }, indent=2), encoding="utf-8")
    print(f"FACE_PRIORITY_MASK_DONE regions={list(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
