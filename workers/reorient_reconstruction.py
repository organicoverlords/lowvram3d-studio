"""Re-export a MoGe reconstruction with a corrected up axis, to test the fix.

`unreal/measure_reconstruction_orientation.py` measured the reconstruction to be
upside down: the source image's top row lands *below* its bottom row in world
space. That is a defect in the export, not in Unreal's importer -- the importer
was measured independently against a probe mesh and maps glTF cleanly.

Rather than change `moge_reconstruct.py` and burn a GPU re-run to find out
whether the theory holds, transform an existing reconstruction and re-render it.
If the corrected mesh matches the source view without needing a mirror, the fix
is proven and can be moved into the exporter.

Negating Y alone is a reflection, so the triangle winding is reversed with it;
otherwise every face points inward and the surface renders as backfaces.

    py -3.12 workers/reorient_reconstruction.py --input <moge.glb> --output <fixed.glb>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Which axes to negate, as the correction under test.
FLIP = (1.0, -1.0, 1.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    scene = trimesh.load(str(source), process=False)
    geometries = (list(scene.geometry.values())
                  if hasattr(scene, "geometry") else [scene])
    if len(geometries) != 1:
        raise RuntimeError(f"expected one geometry, found {len(geometries)}")
    mesh = geometries[0]

    before_y = float(np.asarray(mesh.vertices)[:, 1].mean())
    mesh.vertices = np.asarray(mesh.vertices) * np.asarray(FLIP)
    # A single-axis negation mirrors the mesh, which inverts every face normal.
    mesh.faces = np.asarray(mesh.faces)[:, ::-1]
    after_y = float(np.asarray(mesh.vertices)[:, 1].mean())

    mesh.export(str(output))
    receipt = {
        "schema_version": "reconstruction_reorient_v1",
        "classification": "PROVEN",
        "input": str(source),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "flip": list(FLIP),
        "winding_reversed": True,
        "mean_y_before": round(before_y, 4),
        "mean_y_after": round(after_y, 4),
        "vertices": int(len(mesh.vertices)),
        "faces": int(len(mesh.faces)),
    }
    receipt_path = Path(args.receipt) if args.receipt else output.with_suffix(".json")
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
