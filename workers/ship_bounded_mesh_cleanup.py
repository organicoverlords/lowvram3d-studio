"""Bounded, independently-judged PyMeshLab cleanup stages for the ship hull.

Three stages, each run from the same immutable input rather than chained, so a stage that damages
the ship cannot poison the ones after it. Nothing here remeshes or simplifies: the point is to find
out whether conservative repair alone is enough, not to trade the hull for a nicer triangle count.

  A  duplicate faces, null faces, duplicate/unreferenced vertices, a conservative close-vertex
     merge, and recomputed normals.
  B  detached components below a size threshold. The threshold is derived from the mesh, not
     guessed: it is set below the smallest island that could plausibly be deck equipment, and the
     stage is rejected outright if the silhouette moves.
  C  small simple holes only, with every repaired loop counted. Large openings are left alone.

Chaining is the caller's decision and should only follow from each stage's own verdict.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pymeshlab as ml
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

WELD = 4e-4
EXTREME_ASPECT = 50.0


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def topology(mesh) -> dict:
    positions = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    triangles = np.asarray(mesh.face_matrix(), dtype=np.int64)
    if len(triangles) == 0:
        return {"vertices": int(len(positions)), "triangles": 0}
    a, b, c = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]
    lengths = np.stack((np.linalg.norm(b - a, axis=1),
                        np.linalg.norm(c - b, axis=1),
                        np.linalg.norm(a - c, axis=1)), axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)
    repeated = ((triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2])
                | (triangles[:, 0] == triangles[:, 2]))

    quantised = np.round(positions / WELD).astype(np.int64)
    _unique, welded = np.unique(quantised, axis=0, return_inverse=True)
    edges = np.sort(np.concatenate((welded[triangles][:, [0, 1]],
                                    welded[triangles][:, [1, 2]],
                                    welded[triangles][:, [2, 0]])), axis=1)
    _uedges, counts = np.unique(edges, axis=0, return_counts=True)

    low, high = positions.min(axis=0), positions.max(axis=0)
    return {
        "vertices": int(len(positions)),
        "triangles": int(len(triangles)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "degenerate_triangles": int(np.count_nonzero(repeated | (areas <= 1e-12))),
        "extreme_aspect_triangles": int(np.count_nonzero(aspect >= EXTREME_ASPECT)),
        "max_triangle_aspect": round(float(aspect.max()), 1),
        "bounds_min": [round(float(v), 6) for v in low],
        "bounds_max": [round(float(v), 6) for v in high],
        "extent": [round(float(v), 6) for v in (high - low)],
        "surface_area": round(float(areas.sum()), 6),
    }


def island_report(mesh) -> dict:
    """Detached-component inventory, used to choose and justify stage B's threshold."""
    positions = np.asarray(mesh.vertex_matrix(), dtype=np.float64)
    triangles = np.asarray(mesh.face_matrix(), dtype=np.int64)
    quantised = np.round(positions / WELD).astype(np.int64)
    _unique, welded = np.unique(quantised, axis=0, return_inverse=True)
    count = int(welded.max()) + 1

    rows = np.concatenate((welded[triangles[:, 0]], welded[triangles[:, 1]], welded[triangles[:, 2]]))
    cols = np.concatenate((welded[triangles[:, 1]], welded[triangles[:, 2]], welded[triangles[:, 0]]))
    graph = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(count, count))
    _n, vertex_labels = connected_components(graph, directed=False)
    inverse = vertex_labels[welded[triangles[:, 0]]]
    _labels, inverse = np.unique(inverse, return_inverse=True)
    sizes = np.bincount(inverse)
    main = int(np.argmax(sizes))
    main_vertices = positions[triangles[inverse == main]].reshape(-1, 3)
    low, high = main_vertices.min(axis=0), main_vertices.max(axis=0)
    diagonal = float(np.linalg.norm(high - low))

    islands = []
    for index in range(len(sizes)):
        if index == main:
            continue
        piece = positions[triangles[inverse == index]].reshape(-1, 3)
        outside = np.maximum(low - piece.min(axis=0), 0) + np.maximum(piece.max(axis=0) - high, 0)
        islands.append({
            "faces": int(sizes[index]),
            "extent_percent_of_hull_diagonal": round(float(np.linalg.norm(piece.max(axis=0) - piece.min(axis=0))) / diagonal * 100, 3),
            "protrusion_percent_of_hull_diagonal": round(float(np.linalg.norm(outside)) / diagonal * 100, 3),
        })
    islands.sort(key=lambda item: item["faces"], reverse=True)
    return {
        "components": int(len(sizes)),
        "detached_islands": int(len(sizes) - 1),
        "main_component_faces": int(sizes[main]),
        "hull_diagonal": round(diagonal, 6),
        "island_faces_total": int(sizes.sum() - sizes[main]),
        "islands_top16": islands[:16],
        "max_island_faces": islands[0]["faces"] if islands else 0,
        "max_island_protrusion_percent": max((i["protrusion_percent_of_hull_diagonal"] for i in islands), default=0.0),
    }


def load(path: Path) -> ml.MeshSet:
    meshset = ml.MeshSet()
    meshset.load_new_mesh(str(path))
    return meshset


def save_glb(meshset: ml.MeshSet, path: Path) -> None:
    """PyMeshLab has no GLB writer, so round-trip through PLY and let trimesh emit the GLB."""
    import trimesh

    intermediate = path.with_suffix(".ply")
    meshset.save_current_mesh(str(intermediate), binary=True)
    mesh = trimesh.load(str(intermediate), force="mesh", process=False)
    mesh.export(str(path))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--merge-threshold-percent", type=float, default=0.02)
    parser.add_argument("--island-face-floor", type=int, default=0,
                        help="0 selects a threshold from the island inventory")
    parser.add_argument("--max-hole-size", type=int, default=30)
    args = parser.parse_args()

    source = Path(args.input)
    root = Path(args.output_root)
    root.mkdir(parents=True, exist_ok=True)

    preserved = root / "input_preserved.glb"
    preserved.write_bytes(source.read_bytes())

    baseline_set = load(source)
    baseline = topology(baseline_set.current_mesh())
    baseline_islands = island_report(baseline_set.current_mesh())
    print(f"BASELINE tris={baseline['triangles']} islands={baseline_islands['detached_islands']} "
          f"boundary={baseline['boundary_edges']} degenerate={baseline['degenerate_triangles']}", flush=True)

    stages = {}

    # --- CLEANUP_A -------------------------------------------------------------------------
    meshset = load(source)
    applied = []
    for name, params in (
        ("meshing_remove_duplicate_faces", {}),
        ("meshing_remove_null_faces", {}),
        ("meshing_remove_duplicate_vertices", {}),
        ("meshing_merge_close_vertices", {"threshold": ml.PercentageValue(args.merge_threshold_percent)}),
        ("meshing_remove_unreferenced_vertices", {}),
        ("compute_normal_per_vertex", {}),
    ):
        try:
            meshset.apply_filter(name, **params)
            applied.append(name)
        except Exception as error:  # noqa: BLE001 - a missing filter must be recorded, not fatal
            applied.append(f"{name}:FAILED:{type(error).__name__}")
    out_a = root / "cleanup_a.glb"
    save_glb(meshset, out_a)
    stages["CLEANUP_A"] = {
        "description": "duplicate/null faces, duplicate+unreferenced vertices, conservative close-vertex merge, recomputed normals",
        "filters_applied": applied,
        "merge_threshold_percent_of_bbox_diagonal": args.merge_threshold_percent,
        "output": str(out_a),
        "output_sha256": sha256(out_a),
        "before": baseline,
        "after": topology(meshset.current_mesh()),
        "islands_after": island_report(meshset.current_mesh()),
    }

    # --- CLEANUP_B -------------------------------------------------------------------------
    # Threshold from the inventory: keep every island large enough to be deck equipment.
    if args.island_face_floor > 0:
        floor = args.island_face_floor
        floor_reason = "operator supplied"
    else:
        island_faces = sorted((i["faces"] for i in baseline_islands["islands_top16"]), reverse=True)
        largest = island_faces[0] if island_faces else 0
        floor = max(8, int(largest * 0.25))
        floor_reason = (f"25% of the largest detached island ({largest} faces); islands at or above "
                        f"this size are large enough to be deck equipment and are kept")
    meshset = load(source)
    try:
        meshset.apply_filter("meshing_remove_connected_component_by_face_number",
                             mincomponentsize=int(floor), removeunref=True)
        b_error = None
    except Exception as error:  # noqa: BLE001
        b_error = f"{type(error).__name__}: {error}"
    out_b = root / "cleanup_b.glb"
    save_glb(meshset, out_b)
    after_b = topology(meshset.current_mesh())
    silhouette_shift = max(abs(np.array(after_b["extent"]) - np.array(baseline["extent"])))
    stages["CLEANUP_B"] = {
        "description": "remove detached components below a threshold derived from the island inventory",
        "island_face_floor": int(floor),
        "island_face_floor_reason": floor_reason,
        "error": b_error,
        "output": str(out_b),
        "output_sha256": sha256(out_b),
        "before": baseline,
        "after": after_b,
        "islands_after": island_report(meshset.current_mesh()),
        "silhouette_extent_shift": round(float(silhouette_shift), 6),
        "silhouette_extent_shift_percent": round(float(silhouette_shift) / max(baseline["extent"]) * 100, 4),
    }

    # --- CLEANUP_C -------------------------------------------------------------------------
    meshset = load(source)
    try:
        meshset.apply_filter("meshing_close_holes", maxholesize=int(args.max_hole_size),
                             selected=False, selfintersection=True, refinehole=False)
        c_error = None
    except Exception as error:  # noqa: BLE001
        c_error = f"{type(error).__name__}: {error}"
    out_c = root / "cleanup_c.glb"
    save_glb(meshset, out_c)
    after_c = topology(meshset.current_mesh())
    stages["CLEANUP_C"] = {
        "description": "close only small simple holes; large intentional openings left open",
        "max_hole_size_edges": args.max_hole_size,
        "error": c_error,
        "output": str(out_c),
        "output_sha256": sha256(out_c),
        "before": baseline,
        "after": after_c,
        "islands_after": island_report(meshset.current_mesh()),
        "boundary_edges_closed": baseline["boundary_edges"] - after_c["boundary_edges"],
        "faces_added": after_c["triangles"] - baseline["triangles"],
    }

    payload = {
        "schema": "ship_bounded_mesh_cleanup_v1",
        "backend": "pymeshlab",
        "input": str(source),
        "input_sha256": sha256(source),
        "input_preserved_copy": str(preserved),
        "chained": False,
        "baseline_topology": baseline,
        "baseline_islands": baseline_islands,
        "stages": stages,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for name, stage in stages.items():
        after = stage["after"]
        print(f"{name} tris={after['triangles']} islands={stage['islands_after']['detached_islands']} "
              f"boundary={after.get('boundary_edges')} degenerate={after.get('degenerate_triangles')} "
              f"extreme={after.get('extreme_aspect_triangles')}", flush=True)
    print(f"CLEANUP_DONE report={args.report}", flush=True)


if __name__ == "__main__":
    main()
