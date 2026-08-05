"""Flatten a generated mesh's surface noise without rounding off its features.

Hunyuan3D output has a characteristic defect visible in every unlit preview this
project has rendered: broad surfaces that should be flat are covered in low
shallow lumps, as though the model were wet clay. On the barn it destroyed the
subject -- a building is defined by planes meeting at creases, and it had
neither. On the riverboat it is milder but still there, worst on the top deck,
which is a field of blisters.

Ordinary Laplacian smoothing cannot fix this. It moves every vertex toward its
neighbours regardless of whether it sits on noise or on an edge, so the lumps
flatten and the ornament flattens with them.

Two-step smoothing separates the two. First it averages face normals, but only
across pairs whose normals already agree to within a threshold; a crease
sharper than that is left alone. Then it repositions vertices to fit the
smoothed normal field. Noise, whose normals vary slightly, is averaged away.
Features, whose normals differ by more than the threshold, survive.

The threshold is therefore the whole decision, and it is a property of the
subject rather than of the algorithm: what counts as a feature on an ornate
paddle steamer is not what counts on a barn wall. It is swept and looked at, not
guessed.

    py -3.12 workers/refine_mesh_surface.py --input lod.glb --output clean.glb \\
        --normal-threshold 35
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

# Degrees. Face-normal pairs closer than this are averaged together; anything
# sharper is treated as a feature and preserved. MeshLab's default of 60 is
# permissive enough to swallow shallow ornament, which is most of what makes a
# generated asset readable.
DEFAULT_NORMAL_THRESHOLD = 35.0
DEFAULT_SMOOTH_STEPS = 3


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--normal-threshold", type=float,
                        default=DEFAULT_NORMAL_THRESHOLD)
    parser.add_argument("--smooth-steps", type=int, default=DEFAULT_SMOOTH_STEPS)
    parser.add_argument("--normal-steps", type=int, default=20)
    parser.add_argument("--fit-steps", type=int, default=20)
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import pymeshlab

    source = Path(args.input).resolve()
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    meshes = pymeshlab.MeshSet()
    meshes.load_new_mesh(str(source))
    before = meshes.current_mesh()
    before_vertices = before.vertex_number()
    before_faces = before.face_number()
    before_points = np.asarray(before.vertex_matrix(), dtype=np.float64).copy()

    meshes.apply_coord_two_steps_smoothing(
        stepsmoothnum=args.smooth_steps,
        normalthr=args.normal_threshold,
        stepnormalnum=args.normal_steps,
        stepfitnum=args.fit_steps)

    after = meshes.current_mesh()
    after_points = np.asarray(after.vertex_matrix(), dtype=np.float64)
    # How far the surface actually moved, in units of the model's own size.
    # A refinement that shifts vertices by a noticeable fraction of the subject
    # is not cleaning noise, it is reshaping -- and that must be visible as a
    # number, not discovered later in a render.
    extent = float(np.linalg.norm(before_points.max(axis=0) - before_points.min(axis=0)))
    displacement = np.linalg.norm(after_points - before_points, axis=1)

    # pymeshlab cannot write glTF, so the vertices come back through trimesh.
    # Only positions changed -- two-step smoothing moves vertices and touches
    # neither the face list nor the vertex count -- so rebuilding from the
    # original faces is exact rather than a re-import.
    import trimesh

    refined = trimesh.Trimesh(
        vertices=after_points,
        faces=np.asarray(after.face_matrix(), dtype=np.int64),
        process=False)
    refined.export(str(output))

    receipt = {
        "schema_version": "mesh_surface_refine_v1",
        "classification": "PROVEN",
        "input": str(source),
        "output": str(output),
        "output_bytes": output.stat().st_size,
        "filter": "apply_coord_two_steps_smoothing",
        "normal_threshold_deg": args.normal_threshold,
        "smooth_steps": args.smooth_steps,
        "vertices": before_vertices,
        "faces": before_faces,
        "topology_changed": bool(after.vertex_number() != before_vertices
                                 or after.face_number() != before_faces),
        "diagonal_extent": round(extent, 4),
        "mean_displacement_fraction_of_extent": round(
            float(displacement.mean() / max(extent, 1e-9)), 6),
        "max_displacement_fraction_of_extent": round(
            float(displacement.max() / max(extent, 1e-9)), 6),
    }
    receipt_path = (Path(args.receipt) if args.receipt
                    else output.with_suffix(".refine.json"))
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                            encoding="utf-8")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
