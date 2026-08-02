"""Export image-world arrays as a diagnostic Unreal Landscape package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import traceback

import cv2
import numpy as np

from lowvram3d.image_world.unreal_landscape import (
    encode_unreal_heightmap,
    encode_weightmap,
    landscape_xy_scale_cm,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--world-width-m", type=float, default=4096.0)
    args = parser.parse_args()

    projection = Path(args.projection).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "unreal-landscape-manifest.json"

    try:
        arrays = projection / "arrays"
        height = np.load(arrays / "completed-height.npy", allow_pickle=False).astype(np.float32)
        observed = np.load(arrays / "observed-mask.npy", allow_pickle=False).astype(bool)
        generated = np.load(arrays / "generated-mask.npy", allow_pickle=False).astype(bool)
        confidence = np.load(arrays / "confidence.npy", allow_pickle=False).astype(np.float32)

        encoding = encode_unreal_heightmap(height)
        xy_scale = landscape_xy_scale_cm(args.world_width_m, height.shape[0])
        height_path = output / "heightmap_r16.png"
        observed_path = output / "weight_observed.png"
        generated_path = output / "weight_generated.png"
        confidence_path = output / "weight_confidence.png"

        _write(height_path, encoding.encoded)
        _write(observed_path, encode_weightmap(observed.astype(np.float32)))
        _write(generated_path, encode_weightmap(generated.astype(np.float32)))
        _write(confidence_path, encode_weightmap(confidence))

        manifest = {
            "status": "UNREAL_DIAGNOSTIC_PACKAGE_BUILT",
            "classification": "NOT_READY_FOR_AUTOMATIC_IMPORT",
            "promotion_allowed": False,
            "heightmap": height_path.name,
            "weightmaps": {
                "observed": observed_path.name,
                "generated": generated_path.name,
                "confidence": confidence_path.name,
            },
            "landscape": {
                "vertex_count": int(height.shape[0]),
                "world_width_m": args.world_width_m,
                "xy_scale_cm": xy_scale,
                "z_scale": encoding.landscape_z_scale,
                "actor_z_cm": encoding.actor_z_cm,
                "source_minimum_m": encoding.source_minimum_m,
                "source_maximum_m": encoding.source_maximum_m,
                "quantization_step_m": encoding.quantization_step_m,
            },
            "requirements_before_import": [
                "semantic terrain separation",
                "source-camera visual review",
                "water-level validation",
                "landmark and residual-mesh extraction",
            ],
        }
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("IMAGE_WORLD_UNREAL_DIAGNOSTIC_PACKAGE_BUILT")
        return 0
    except Exception as exc:
        manifest_path.write_text(
            json.dumps(
                {
                    "status": "FAILED",
                    "promotion_allowed": False,
                    "errors": [f"{type(exc).__name__}: {exc}", traceback.format_exc()],
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return 2


def _write(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"OpenCV failed to write {path}")
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError(f"output is missing or empty: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
