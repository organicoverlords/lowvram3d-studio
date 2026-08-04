"""Mirror a reconstruction's Y axis. Kept as a diagnostic, not as a repair.

**This does not fix the reconstruction, and applying it produces a mesh that is
wrong in world space.** It is checked in because running it is what exposed the
real defect, and because that lesson is worth being able to reproduce.

The story: a reconstruction rendered upside down, so this was written to flip
its geometry, and the flipped mesh scored 1.439 against the source view where
the original scored 0.301. That looked conclusive and was not. The texture was
the thing that was inverted -- trimesh flips v on glTF export, so authoring v in
image-row order paints the sky along the ground -- and flipping the geometry
merely made a second wrong cancel the first *from the source camera*. The higher
score came from the ground being painted with bright cloud, which is what the
source's upper region also contains.

The defect is fixed in `moge_reconstruct.py`, at the UVs. See
`docs/AXIS_CONVENTIONS.md`, and settle questions like this by reading the GLB's
accessors rather than by scoring a render.

Negating one axis is a reflection, so triangle winding is reversed with it.

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
