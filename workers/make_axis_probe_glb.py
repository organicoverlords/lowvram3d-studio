"""Write a glTF probe whose three axes are individually identifiable after import.

The glTF->Unreal axis mapping has been guessed three times in this project and
was wrong at least once, so settle it by measurement instead. The probe carries
one small box per axis at a *different distance* along that axis:

    +X at 1 m,  +Y at 2 m,  +Z at 3 m

Distinct magnitudes make the recovered mapping unambiguous in both axis and
sign: whatever Unreal reports at ~200 cm came from glTF +Y, and its sign says
whether the importer flipped it. A fourth box sits at the origin so a mesh that
imported without any transform is still distinguishable from one that did not
import at all.

    python workers/make_axis_probe_glb.py --output evidence/axis-probe/probe.glb
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# (name, centre in glTF metres). Magnitudes differ so each is identifiable.
MARKERS = [
    ("origin", (0.0, 0.0, 0.0)),
    ("gltf_x_plus", (1.0, 0.0, 0.0)),
    ("gltf_y_plus", (0.0, 2.0, 0.0)),
    ("gltf_z_plus", (0.0, 0.0, 3.0)),
]
HALF_EXTENT = 0.05  # 10 cm cubes: small enough that clusters never overlap.


def build():
    import numpy as np
    import trimesh

    meshes = []
    for _name, centre in MARKERS:
        box = trimesh.creation.box(extents=[HALF_EXTENT * 2] * 3)
        box.apply_translation(np.asarray(centre, dtype=float))
        meshes.append(box)
    return trimesh.util.concatenate(meshes)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    mesh = build()
    mesh.export(str(output))

    receipt = {
        "schema_version": "axis_probe_v1",
        "classification": "PROVEN",
        "glb": str(output),
        "glb_bytes": output.stat().st_size,
        "units": "metres",
        "half_extent_m": HALF_EXTENT,
        "markers": {name: list(centre) for name, centre in MARKERS},
        "vertices": int(len(mesh.vertices)),
    }
    receipt_path = Path(args.receipt) if args.receipt else output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
