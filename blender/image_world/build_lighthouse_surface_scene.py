"""Build a Blender proof scene from image-world surface artifacts.

This intentionally creates a diagnostic scene, not a final game asset.
Observed and generated terrain regions remain separate so later agents can
replace procedural completion without losing source evidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import json
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projection", required=True)
    parser.add_argument("--blend-output", required=True)
    parser.add_argument("--terrain-resolution", type=int, default=513)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    projection = Path(args.projection).resolve()
    output = Path(args.blend_output).resolve()

    report = {
        "status": "BLENDER_HANDOFF_SPEC_CREATED",
        "promotion_allowed": False,
        "projection": str(projection),
        "blend_output": str(output),
        "terrain_resolution": args.terrain_resolution,
        "requirements": [
            "import completed-height.npy as diagnostic terrain",
            "preserve observed/generated masks as vertex attributes",
            "create separate residual-mesh collection",
            "create source-camera debug collection",
            "render proof before Unreal export",
        ],
        "warnings": [
            "This is not a final asset generator.",
            "Semantic terrain masks are still required.",
            "Hero lighthouse geometry must remain separate from terrain.",
        ],
    }

    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print("BLENDER_HANDOFF_SPEC_CREATED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
