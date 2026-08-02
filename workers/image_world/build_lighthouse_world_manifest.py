"""Create a machine-readable handoff from MoGe observations to DCC/engine stages.

The manifest is deliberately a contract document. It does not claim that an
image has become a finished world; it records what was observed and what still
requires procedural generation or artist/agent decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from datetime import datetime, timezone


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--surface", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    geometry = Path(args.geometry).resolve()
    surface = Path(args.surface).resolve()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)

    required_geometry = ["points.npy", "depth.npy", "normal.npy", "mask.npy", "intrinsics.npy"]
    required_surface = [
        "arrays/completed-height.npy",
        "arrays/slope-degrees.npy",
        "arrays/stream-mask.npy",
        "surface-projection-report.json",
    ]

    manifest = {
        "status": "READY_FOR_DCC_RECONSTRUCTION",
        "classification": "IMAGE_WORLD_OBSERVATION_NOT_FINAL_WORLD",
        "source_contract": {
            "geometry_files": {
                name: {
                    "exists": (geometry / name).is_file(),
                    "sha256": sha256(geometry / name) if (geometry / name).is_file() else None,
                }
                for name in required_geometry
            },
            "surface_files": {
                name: {
                    "exists": (surface / name).is_file(),
                }
                for name in required_surface
            },
        },
        "next_agents": {
            "blender": [
                "create terrain mesh from completed-height.npy",
                "scatter procedural vegetation using confidence and slope",
                "generate shoreline/cliff materials from masks",
            ],
            "unreal": [
                "import heightfield only after semantic terrain validation",
                "use PCG for foliage and environment dressing",
                "keep observed/generated boundary metadata",
            ],
        },
        "not_proven": [
            "camera calibration accuracy",
            "semantic terrain segmentation",
            "architectural reconstruction",
            "photorealistic material recovery",
        ],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    (output / "lighthouse-world-manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("IMAGE_WORLD_MANIFEST_READY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
