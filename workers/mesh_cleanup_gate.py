"""Bring a raw generated mesh up to the topology the hosted services ship.

Three hosted outputs of the Lucky Drown were measured against our best TRELLIS
seed, and the interesting number was not proportion:

                     shells   main-area   winding   watertight   faces
  online A                1      1.0000      True         True   499,974
  online B                1      1.0000      True         True   499,894
  online C (low poly)     1      1.0000      True         True    49,794
  TRELLIS seed 1006  51,502      0.7548     False        False 2,035,434

All three services ship one welded, watertight, consistently wound shell inside
a face budget. We ship raw marching-cubes output with a quarter of its surface
area off the main body. That is the whole remaining gap, and it is a cleanup
problem rather than a generation problem.

Order matters and is taken from TRELLIS.2's own postprocess (weld, then clean,
then decimate). Welding first is not cosmetic: marching-cubes output arrives
unwelded, so connectivity is meaningless until it runs and every component
count taken before it is a count of triangles. Debris removal comes before
decimation because decimating debris spends budget on geometry that is about to
be deleted -- and because decimation on this machine has stalled at 1.13-1.17M
faces before, so going in with a quarter fewer faces is the difference between
finishing and being killed.

    py workers/mesh_cleanup_gate.py --mesh raw.glb --out clean.glb \
       --target-faces 500000

The gate reports rather than asserts. A mesh that comes out non-watertight is
still written, with the receipt saying so -- hole filling on a generated mesh
can close a doorway as easily as a crack, and silently "repairing" a subject
into a solid block is worse than shipping a known-open one.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

#: Components smaller than this fraction of the largest component's area are
#: debris: speckle, interior shards, and the disconnected fragments that a
#: 0.5-percentile extent trim was invented to hide. Area rather than face count,
#: because a dense speck has many faces and covers nothing.
DEBRIS_AREA_FRACTION = 0.001

#: What the hosted services ship. Not a hard requirement, a default.
DEFAULT_TARGET_FACES = 500_000


def load(path: Path):
    import trimesh
    scene = trimesh.load(path, process=False)
    return scene.to_geometry() if hasattr(scene, "geometry") else scene


def components(mesh):
    """Per-face component labels and their areas, without building submeshes.

    `mesh.split()` is the obvious call and it is a machine-killer here. It
    materialises one full Trimesh per component, and the raw seed has 51,502 of
    them across 2M faces; called once per pipeline step on a 16 GB box that took
    the whole PC down twice. Labelling the face-adjacency graph returns index
    arrays instead, so the cost is O(faces) and nothing is copied.
    """
    import numpy as np
    from trimesh.graph import connected_components

    labels = np.full(len(mesh.faces), -1, dtype=np.int64)
    groups = connected_components(mesh.face_adjacency,
                                  nodes=np.arange(len(mesh.faces)))
    for index, group in enumerate(groups):
        labels[group] = index
    areas = np.bincount(labels, weights=mesh.area_faces,
                        minlength=max(len(groups), 1))
    return labels, areas


def audit(mesh) -> dict:
    import numpy as np
    _, areas = components(mesh)
    total = float(areas.sum()) or 1.0
    return {
        "faces": int(len(mesh.faces)),
        "vertices": int(len(mesh.vertices)),
        "shells": int(len(areas)),
        "main_shell_area_fraction": round(float(areas.max() / total), 4),
        "winding_consistent": bool(mesh.is_winding_consistent),
        "watertight": bool(mesh.is_watertight),
    }


def run(mesh_path: Path, out_path: Path, target_faces: int,
        fill_holes: bool = True) -> dict:
    import numpy as np
    import trimesh

    started = time.time()
    mesh = load(mesh_path)
    steps = []

    # 1. Weld. Generated meshes arrive with per-corner vertices -- marching
    #    cubes emits them, and glTF export splits them again on normal and UV
    #    seams. Until this runs, connectivity is fiction: the same mesh reads as
    #    one shell or as half a million depending only on how it was stored.
    mesh.merge_vertices(merge_tex=True, merge_norm=True)
    steps.append({"step": "weld", **audit(mesh)})

    # 2. Remove the geometry that is not geometry: NaN vertices, zero-area
    #    faces, exact duplicates. These break every later stage rather than
    #    merely dirtying it.
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_infinite_values()
    mesh.remove_unreferenced_vertices()
    steps.append({"step": "remove_degenerate", **audit(mesh)})

    # 3. Drop debris, keeping every component that is a real part of the
    #    subject. Not "keep the largest": a paddle wheel can legitimately be a
    #    separate shell, and throwing it away to win a metric would be the
    #    worst possible outcome here.
    labels, areas = components(mesh)
    kept = np.nonzero(areas >= areas.max() * DEBRIS_AREA_FRACTION)[0]
    dropped = len(areas) - len(kept)
    if dropped:
        # Mask faces rather than concatenating submeshes: same result, and it
        # never holds more than one copy of the mesh in memory.
        mesh.update_faces(np.isin(labels, kept))
        mesh.remove_unreferenced_vertices()
    steps.append({"step": "drop_debris", **audit(mesh),
                  "components_kept": int(len(kept)),
                  "components_dropped": int(dropped)})

    # 4. Winding and normals. TRELLIS output is inconsistently wound, which
    #    renders black and breaks texture baking; it is a repair, not a defect.
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_inversion(mesh)
    trimesh.repair.fix_normals(mesh)
    steps.append({"step": "repair_winding", **audit(mesh)})

    if fill_holes:
        # trimesh.repair.fill_holes only closes small triangular gaps, and this
        # surface is not in that regime: the cleaned seed carries 67,616
        # boundary edges and 46,663 non-manifold edges, roughly 9 percent of
        # its edges, against 0 and 0 for every hosted reference. So run it,
        # then hand what remains to manifold3d, which resolves self-intersection
        # and non-manifold structure by construction rather than by patching.
        trimesh.repair.fill_holes(mesh)
        steps.append({"step": "fill_holes", **audit(mesh)})
        try:
            import manifold3d
            solid = manifold3d.Manifold(manifold3d.Mesh(
                vert_properties=np.asarray(mesh.vertices, dtype=np.float32),
                tri_verts=np.asarray(mesh.faces, dtype=np.uint32)))
            built = solid.to_mesh()
            rebuilt = trimesh.Trimesh(np.asarray(built.vert_properties)[:, :3],
                                      np.asarray(built.tri_verts), process=False)
            # manifold3d returns an EMPTY mesh rather than raising when it
            # cannot make sense of the input, and it did exactly that here.
            # Without this guard the stage silently replaces a good 500k mesh
            # with nothing and the gate reports it as a completed step.
            # A repair that deletes the subject is not a repair.
            if len(rebuilt.faces) < len(mesh.faces) * 0.5:
                steps.append({"step": "manifold", "rejected": True,
                              "faces_returned": int(len(rebuilt.faces)),
                              "faces_in": int(len(mesh.faces)),
                              "reason": "returned under half the input; kept previous mesh"})
            else:
                mesh = rebuilt
                steps.append({"step": "manifold", **audit(mesh)})
        except Exception as error:
            # Recorded rather than swallowed: a mesh this stage could not close
            # is a real result about the generator, not a missing feature.
            steps.append({"step": "manifold", "error": f"{type(error).__name__}: {error}"[:200]})

    # 5. Decimate last, on a mesh that no longer wastes budget on debris.
    #    open3d's quadric decimation is the one implementation present in both
    #    interpreters on this machine; pymeshlab and fast_simplification are
    #    not installed, and open3d has stalled here above roughly 1.17M faces,
    #    which is why steps 1-4 come first.
    if target_faces and len(mesh.faces) > target_faces:
        import open3d as o3d
        source = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(mesh.vertices)),
            o3d.utility.Vector3iVector(np.asarray(mesh.faces)))
        reduced = source.simplify_quadric_decimation(int(target_faces))
        mesh = trimesh.Trimesh(np.asarray(reduced.vertices),
                               np.asarray(reduced.triangles), process=False)
        mesh.merge_vertices(merge_tex=True, merge_norm=True)
        trimesh.repair.fix_winding(mesh)
        steps.append({"step": "decimate", **audit(mesh)})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(out_path)

    final = audit(mesh)
    return {
        "schema": "lowvram3d_mesh_cleanup_gate_v1",
        "mesh_in": str(mesh_path),
        "mesh_out": str(out_path),
        "target_faces": target_faces,
        "seconds": round(time.time() - started, 1),
        "before": steps[0] if steps else None,
        "steps": steps,
        "after": final,
        # Stated, not enforced. The hosted references pass all four.
        "meets_hosted_standard": bool(
            final["shells"] == 1 and final["watertight"]
            and final["winding_consistent"]
            and (not target_faces or final["faces"] <= target_faces)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mesh", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--target-faces", type=int, default=DEFAULT_TARGET_FACES)
    parser.add_argument("--no-fill-holes", action="store_true",
                        help="hole filling can close a doorway as well as a crack")
    args = parser.parse_args(argv)

    result = run(args.mesh, args.out, args.target_faces,
                 fill_holes=not args.no_fill_holes)
    args.out.with_suffix(".cleanup.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for step in result["steps"]:
        print("%-18s faces %9d  shells %6d  main-area %.4f  winding %-5s  watertight %s"
              % (step["step"], step["faces"], step["shells"],
                 step["main_shell_area_fraction"], step["winding_consistent"],
                 step["watertight"]))
    print("meets_hosted_standard:", result["meets_hosted_standard"],
          " in", result["seconds"], "s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
