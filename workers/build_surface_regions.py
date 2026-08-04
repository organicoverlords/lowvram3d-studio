"""Create asset-independent geometric surface regions."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from mesh_io import read_glb


def build_regions(positions: np.ndarray, normals: np.ndarray, triangles: np.ndarray,
                  *, weld: float = 4e-4, min_normal_dot: float = 0.45) -> tuple[np.ndarray, dict]:
    positions = np.asarray(positions, np.float64)
    normals = np.asarray(normals, np.float64)
    triangles = np.asarray(triangles, np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("triangles must have shape [T,3]")
    welded = np.unique(np.round(positions / float(weld)).astype(np.int64), axis=0,
                       return_inverse=True)[1]
    corners = welded[triangles]
    rows = np.concatenate([corners[:, 0], corners[:, 1], corners[:, 2]])
    cols = np.concatenate([corners[:, 1], corners[:, 2], corners[:, 0]])
    vertex_count = int(welded.max()) + 1 if welded.size else 0
    vertex_graph = coo_matrix((np.ones(rows.size, np.uint8), (rows, cols)),
                              shape=(vertex_count, vertex_count)).tocsr()
    vertex_graph = vertex_graph.maximum(vertex_graph.T)
    _, vertex_components = connected_components(vertex_graph, directed=False)
    base = vertex_components[corners[:, 0]] if len(corners) else np.zeros(0, np.int32)

    # Regions are connected through welded vertices but sharp normal breaks are barriers. This
    # is intentionally geometric; material and UV boundaries are supporting metadata only.
    edge_map: dict[tuple[int, int], list[int]] = {}
    for tid, tri in enumerate(corners):
        for a, b in ((tri[0], tri[1]), (tri[1], tri[2]), (tri[2], tri[0])):
            edge_map.setdefault(tuple(sorted((int(a), int(b)))), []).append(tid)
    pair_rows, pair_cols = [], []
    face_normals = normals[triangles].mean(axis=1)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)
    for tids in edge_map.values():
        if len(tids) != 2:
            continue
        a, b = tids
        if float(face_normals[a] @ face_normals[b]) >= float(min_normal_dot):
            pair_rows.extend((a, b)); pair_cols.extend((b, a))
    graph = coo_matrix((np.ones(len(pair_rows), np.uint8), (pair_rows, pair_cols)),
                       shape=(len(triangles), len(triangles))).tocsr()
    region_count, regions = connected_components(graph, directed=False)
    centroids = positions[triangles].mean(axis=1) if len(triangles) else np.zeros((0, 3))
    sizes = np.bincount(regions, minlength=region_count)
    kind = np.full(region_count, "unknown_component", dtype="U32")
    kind[sizes >= 100] = "main_shell"
    kind[(sizes > 1) & (sizes < 100)] = "compact_solid_component"
    kind[sizes == 1] = "thin_or_ornament"
    report = {
        "schema": "surface_regions_v1",
        "triangle_count": int(len(triangles)),
        "welded_vertex_count": vertex_count,
        "connected_component_count": int(np.unique(base).size) if len(base) else 0,
        "surface_region_count": int(region_count),
        "minimum_normal_dot": float(min_normal_dot),
        "region_types": {str(i): str(kind[i]) for i in range(region_count)},
        "centroid_count": int(len(centroids)),
        "deterministic": True,
    }
    return regions.astype(np.int32), report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--weld", type=float, default=4e-4)
    parser.add_argument("--min-normal-dot", type=float, default=0.45)
    args = parser.parse_args()
    positions, normals, _uv, triangles = read_glb(Path(args.mesh))
    regions, report = build_regions(positions, normals, triangles, weld=args.weld,
                                    min_normal_dot=args.min_normal_dot)
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    np.save(out / "surface_region_per_triangle.npy", regions)
    report["mesh"] = str(args.mesh)
    report["output"] = str(out / "surface_region_per_triangle.npy")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
