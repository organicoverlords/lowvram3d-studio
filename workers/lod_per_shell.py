"""Decimate a many-shell, non-watertight mesh without destroying it.

`workers/decimate_mesh.py` runs Blender's quadric decimation over the whole
object. That is correct for a watertight Hunyuan mesh with a dozen bodies and
catastrophic for a TRELLIS one: 932k faces across 7,144 open shells came back
as torn plates with holes through the superstructure, and the body count
collapsed from 7,144 to 916 because whole components had been welded away.

Two things go wrong when a global quadric pass meets this topology.

**Open shells collapse across themselves.** Quadric error is defined from the
planes meeting at a vertex. On a closed surface that is well conditioned. On a
one-sided shell -- a railing, a deck edge, a sail of a paddle wheel -- the
boundary vertices have no opposing plane to constrain them, so the cheapest
collapse is often the one that folds the shell flat. Locking the boundary
removes that option.

**A global error budget is spent in the wrong place.** Ranked by absolute
quadric error, a million tiny collapses on ornament cost less than one collapse
on a large hull plate, so a global pass shreds the detail and leaves the flat
areas dense -- the exact inversion of what an LOD wants. Budgeting per shell,
proportional to its own area, keeps each component recognisable.

So: split into connected shells, protect the ones too small or too thin to
survive simplification at all, decimate the rest independently with boundary
weighting, and reassemble.

    py lod_per_shell.py --input mesh.glb --out lod.glb --target 200000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

#: Shells with fewer faces than this are passed through untouched. A railing
#: post or a wheel spoke is only a few hundred triangles; there is no LOD to be
#: had from it and every collapse risks removing it entirely.
MIN_FACES_TO_SIMPLIFY = 400

#: Shells thinner than this, measured as smallest extent over largest, are also
#: passed through. A rail is a long thin box: quadric error barely notices the
#: short axis, so simplification flattens it into a ribbon and then a line.
MIN_THICKNESS_RATIO = 0.06

#: How strongly boundary edges resist collapse. open3d's default is 1.0, which
#: treats a shell edge like any other; the whole point here is that it is not.
BOUNDARY_WEIGHT = 12.0

#: Never take a single shell below this fraction of its original faces, however
#: the global budget works out. Past roughly this point a shell stops reading as
#: the object it was.
MIN_SHELL_RETENTION = 0.15


def load_flat(path):
    import numpy as np
    import trimesh

    scene = trimesh.load(str(path), process=False)
    if hasattr(scene, "geometry"):
        mesh = (scene.to_geometry() if hasattr(scene, "to_geometry")
                else scene.dump(concatenate=True))
    else:
        mesh = scene
    return trimesh.Trimesh(vertices=np.asarray(mesh.vertices),
                           faces=np.asarray(mesh.faces), process=True)


def simplify_shell(shell, target_faces):
    """Quadric decimation of one shell with its boundary weighted heavily."""
    import numpy as np
    import open3d as o3d

    o3d_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(shell.vertices)),
        o3d.utility.Vector3iVector(np.asarray(shell.faces)))
    simplified = o3d_mesh.simplify_quadric_decimation(
        target_number_of_triangles=int(target_faces),
        boundary_weight=BOUNDARY_WEIGHT)
    vertices = np.asarray(simplified.vertices)
    faces = np.asarray(simplified.triangles)
    if len(faces) == 0:
        return None
    import trimesh
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--target", type=int, default=200_000,
                        help="Approximate total triangles after simplification.")
    parser.add_argument("--receipt", default="")
    args = parser.parse_args(argv)

    import numpy as np
    import trimesh

    mesh = load_flat(args.input)
    shells = mesh.split(only_watertight=False)
    if not len(shells):
        raise SystemExit("NO_COMPONENTS")

    areas = np.array([float(s.area) for s in shells])
    face_counts = np.array([len(s.faces) for s in shells])

    protected = []
    simplifiable = []
    for index, shell in enumerate(shells):
        extents = np.sort(np.asarray(shell.extents))
        thin = extents[0] / max(extents[2], 1e-9) < MIN_THICKNESS_RATIO
        if face_counts[index] < MIN_FACES_TO_SIMPLIFY or thin:
            protected.append(index)
        else:
            simplifiable.append(index)

    protected_faces = int(face_counts[protected].sum()) if protected else 0
    budget = max(args.target - protected_faces, 1000)

    # Distribute the budget by area, not by face count: a dense but small
    # ornament should not claim the share of a large hull plate simply because
    # the generator tessellated it more finely.
    if simplifiable:
        share = areas[simplifiable] / max(areas[simplifiable].sum(), 1e-9)
    else:
        share = np.array([])

    pieces = [shells[i] for i in protected]
    results = []
    for position, index in enumerate(simplifiable):
        original = int(face_counts[index])
        target = int(max(budget * share[position],
                         original * MIN_SHELL_RETENTION))
        if target >= original:
            pieces.append(shells[index])
            results.append({"shell": index, "faces": original,
                            "target": original, "action": "kept"})
            continue
        simplified = simplify_shell(shells[index], target)
        if simplified is None:
            pieces.append(shells[index])
            results.append({"shell": index, "faces": original,
                            "action": "kept (simplify returned empty)"})
            continue
        pieces.append(simplified)
        results.append({"shell": index, "faces": original,
                        "target": target,
                        "result": int(len(simplified.faces)),
                        "action": "simplified"})

    combined = trimesh.util.concatenate(pieces)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.export(str(out))

    receipt = {
        "schema_version": "lod_per_shell_v1",
        "input": str(Path(args.input).resolve()),
        "output": str(out.resolve()),
        "input_triangles": int(len(mesh.faces)),
        "output_triangles": int(len(combined.faces)),
        "shells_total": int(len(shells)),
        "shells_protected": len(protected),
        "shells_simplified": len(simplifiable),
        "protected_faces": protected_faces,
        "boundary_weight": BOUNDARY_WEIGHT,
        "min_faces_to_simplify": MIN_FACES_TO_SIMPLIFY,
        "min_thickness_ratio": MIN_THICKNESS_RATIO,
        "min_shell_retention": MIN_SHELL_RETENTION,
        "note": ("per-shell budgets by area; boundaries weighted so open shells "
                 "cannot collapse across themselves"),
    }
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(receipt, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items()}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
