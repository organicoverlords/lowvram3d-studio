"""Geometry-quality metrics computed from a mesh alone, with no source image and no pairing.

Every measure here is intrinsic to the mesh, which is what makes a frozen reference library useful:
a model whose source picture is long gone still tells us what healthy topology, sane component
structure and real surface detail look like. Nothing is written back - meshes are opened read-only.

The metrics were chosen against the defects that actually shipped: fused anatomy and plate-like
antlers show up as abnormally low curvature energy over the upper body, debris as a long tail of
tiny components, a thickened beak as lost detail concentration in the head band, and collapsed
reconstructions as broken bilateral symmetry.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from mesh_io import read_glb, triangle_components

WELD = 4e-4
SYMMETRY_SAMPLE = 120_000


def _edge_table(tris: np.ndarray, welded: np.ndarray):
    corners = welded[tris]
    edges = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return unique, counts


def canonical_axes(positions: np.ndarray, sample: int = 60_000) -> tuple[int, int, float]:
    """Find which axis is lateral and which is up, without trusting the file's convention.

    Exporters disagree about up: a model authored Z-up and one authored Y-up look identical in a
    viewer that compensates, and nothing in the mesh announces which was meant. Assuming Y-up made
    a reference that is lying down report its flank as its head band, and the resulting comparison
    looked entirely reasonable.

    A character's lateral axis is the one it is most symmetric about, so each axis is tried as the
    mirror normal and the best-scoring one wins. Up is then whichever of the remaining two the
    subject extends furthest along.
    """
    from scipy.spatial import cKDTree

    points = positions
    if len(points) > sample:
        rng = np.random.default_rng(20260801)
        points = points[rng.choice(len(points), size=sample, replace=False)]
    tree = cKDTree(points)
    extent = positions.max(0) - positions.min(0)
    scale = max(float(np.linalg.norm(extent)), 1e-12)

    scores = []
    for axis in range(3):
        centre = float(np.median(points[:, axis]))
        mirrored = points.copy()
        mirrored[:, axis] = 2.0 * centre - mirrored[:, axis]
        scores.append(float(np.median(tree.query(mirrored, k=1)[0])) / scale)
    lateral = int(np.argmin(scores))
    remaining = [a for a in range(3) if a != lateral]
    up = remaining[int(np.argmax(extent[remaining]))]
    return lateral, up, scores[lateral]


def up_axis_sign(positions: np.ndarray, tris: np.ndarray, up: int) -> float:
    """Which end along the up axis is the top. Returns +1 if larger coordinate is up.

    Knowing the axis is not enough: a Z-up export can still be stored either way round, and getting
    it wrong puts the antler bar and staff ring at the bottom of the render while every number still
    looks plausible.

    Decided by where the surface area sits, not by where the silhouette is widest. A standing figure
    carries most of its area low - robe, legs, base - while the head end is comparatively slight.
    Width would be the obvious test and is wrong here: this subject's antler bar is the widest thing
    on it and sits at the very top.
    """
    a, b, c = positions[tris[:, 0]], positions[tris[:, 1]], positions[tris[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    centre = positions[tris].mean(axis=1)[:, up]
    midpoint = 0.5 * (float(positions[:, up].min()) + float(positions[:, up].max()))
    lower = float(area[centre < midpoint].sum())
    upper = float(area[centre >= midpoint].sum())
    return 1.0 if lower >= upper else -1.0


def measure(path: Path, sample: int = SYMMETRY_SAMPLE) -> dict:
    positions, _, _, tris = read_glb(path)
    positions = positions.astype(np.float64)
    lateral_axis, up_axis, _ = canonical_axes(positions)
    up_sign = up_axis_sign(positions, tris, up_axis)
    if up_sign < 0:
        positions = positions.copy()
        positions[:, up_axis] = -positions[:, up_axis]
    lo, hi = positions.min(0), positions.max(0)
    extent = hi - lo
    scale = float(np.linalg.norm(extent))
    height = max(float(extent[up_axis]), 1e-9)

    component, welded = triangle_components(positions, tris, WELD)
    # One representative position per welded vertex, so edge lengths can be taken from the same
    # welded indices the edge table is built on.
    positions_by_vertex = np.zeros((int(welded.max()) + 1, 3), np.float64)
    positions_by_vertex[welded] = positions
    sizes = np.bincount(component)
    body = int(sizes.max())
    total_tris = int(len(tris))

    # ---- topology health
    unique_edges, counts = _edge_table(tris, welded)
    boundary = int((counts == 1).sum())
    non_manifold = int((counts > 2).sum())

    # ---- triangle shape: slivers destroy both baking and decimation
    a = positions[tris[:, 0]]
    b = positions[tris[:, 1]]
    c = positions[tris[:, 2]]
    cross = np.cross(b - a, c - a)
    area = 0.5 * np.linalg.norm(cross, axis=1)
    lengths = np.stack([np.linalg.norm(b - a, axis=1),
                        np.linalg.norm(c - b, axis=1),
                        np.linalg.norm(a - c, axis=1)], axis=1)
    longest = lengths.max(axis=1)
    # Ratio of area to the square of the longest edge; an equilateral triangle sits near 0.43.
    quality = np.where(longest > 1e-12, area / np.maximum(longest ** 2, 1e-18), 0.0)
    sliver_fraction = float((quality < 0.02).mean())

    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)

    # ---- surface smoothness and detail, from dihedral angles across shared edges
    shared = unique_edges[counts == 2]
    curvature_energy = 0.0
    upper_detail = 0.0
    if len(shared):
        corners = welded[tris]
        edge_list = np.concatenate([corners[:, [0, 1]], corners[:, [1, 2]], corners[:, [2, 0]]])
        edge_list = np.sort(edge_list, axis=1)
        owner = np.tile(np.arange(len(tris)), 3)
        order = np.lexsort((edge_list[:, 1], edge_list[:, 0]))
        sorted_edges, sorted_owner = edge_list[order], owner[order]
        same = np.all(sorted_edges[1:] == sorted_edges[:-1], axis=1)
        left = sorted_owner[:-1][same]
        right = sorted_owner[1:][same]
        if len(left):
            dot = np.clip(np.einsum("ij,ij->i", normals[left], normals[right]), -1.0, 1.0)
            angle = np.arccos(dot)
            # Total absolute curvature per unit area, not mean dihedral angle. The bare angle
            # shrinks as triangles get smaller, so a dense mesh scores as "smooth" purely for being
            # dense - besgu at 1.5M triangles read 0.059 against 0.293 for a 178k mesh, which is a
            # measurement of triangle count wearing the name of shape. Weighting each angle by its
            # edge length and dividing by surface area removes the resolution dependence; the
            # factor of `scale` makes it dimensionless so meshes of different physical size compare.
            shared_length = np.linalg.norm(
                positions_by_vertex[sorted_edges[:-1][same][:, 0]]
                - positions_by_vertex[sorted_edges[:-1][same][:, 1]], axis=1)
            surface = float(area.sum())
            curvature_energy = float((angle * shared_length).sum() / max(surface, 1e-12) * scale)
            centroid_height = ((positions[tris].mean(axis=1)[:, up_axis] - lo[up_axis]) / height)
            band = (centroid_height[left] > 0.70) & (centroid_height[right] > 0.70)
            if band.sum() > 32:
                band_area = float(area[left[band]].sum() + area[right[band]].sum()) * 0.5
                upper_detail = float((angle[band] * shared_length[band]).sum()
                                     / max(band_area, 1e-12) * scale)

    # ---- bilateral symmetry about the plane through the centroid, normal to X
    centre_x = float(np.median(positions[:, lateral_axis]))
    try:
        from scipy.spatial import cKDTree
        # The tree gets every vertex; only the queries are sampled. Subsampling both sides made the
        # nearest-neighbour distance a measure of the sampling gap rather than of asymmetry, and
        # every mesh scored exactly 0.
        tree = cKDTree(positions)
        rng = np.random.default_rng(20260801)
        index = (rng.choice(len(positions), size=sample, replace=False)
                 if len(positions) > sample else np.arange(len(positions)))
        queries = positions[index].copy()
        queries[:, lateral_axis] = 2.0 * centre_x - queries[:, lateral_axis]
        distance = tree.query(queries, k=1)[0]
        # Reported as a fraction of the bounding diagonal, no saturating rescale: a raw ratio stays
        # informative at both ends instead of collapsing everything past a threshold onto zero.
        symmetry = float(np.median(distance) / max(scale, 1e-12))
    except Exception:
        symmetry = float("nan")

    # ---- thin features: how much surface sits in structures thinner than a fraction of height
    thin_fraction = float(area[quality < 0.10].sum() / max(area.sum(), 1e-12))

    tiny = int((sizes < max(8, total_tris // 20000)).sum())

    return {
        "mesh": str(path),
        "up_axis": "xyz"[up_axis],
        "up_axis_sign": int(up_sign),
        "lateral_axis": "xyz"[lateral_axis],
        "triangles": total_tris,
        "vertices": int(len(positions)),
        "extent": [round(float(v), 5) for v in extent],
        "axis_ratio": round(float(np.sort(extent)[-1] / max(np.sort(extent)[0], 1e-9)), 4),
        "components": int(len(sizes)),
        "largest_component_fraction": round(body / max(total_tris, 1), 5),
        "tiny_components": tiny,
        "debris_triangle_fraction": round(float(sizes[sizes < max(8, total_tris // 20000)].sum()
                                               / max(total_tris, 1)), 6),
        "boundary_edge_fraction": round(boundary / max(len(unique_edges), 1), 6),
        "non_manifold_edge_fraction": round(non_manifold / max(len(unique_edges), 1), 6),
        "sliver_triangle_fraction": round(sliver_fraction, 6),
        "triangle_area_cv": round(float(area.std() / max(area.mean(), 1e-18)), 4),
        "curvature_energy": round(curvature_energy, 6),
        "upper_band_detail": round(upper_detail, 6),
        "symmetry_median_distance_fraction": round(symmetry, 6),
        "thin_feature_fraction": round(thin_fraction, 5),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", action="append", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    results, failures = [], []
    for entry in args.mesh:
        path = Path(entry)
        if not path.is_file() or path.suffix.lower() != ".glb":
            failures.append({"mesh": str(path), "reason": "not a readable .glb"})
            continue
        try:
            record = measure(path)
            record["label"] = args.label
            results.append(record)
            print(f"QUALITY {path.name}: tris={record['triangles']} "
                  f"comp={record['components']} "
                  f"asym={record['symmetry_median_distance_fraction']} "
                  f"curv={record['curvature_energy']} upper={record['upper_band_detail']}",
                  flush=True)
        except Exception as error:  # noqa: BLE001 - a bad reference must not stop the sweep
            failures.append({"mesh": str(path), "reason": str(error)[:300]})
            print(f"QUALITY_FAILED {path.name}: {str(error)[:160]}", flush=True)

    report = {"label": args.label, "measured": len(results), "failed": len(failures),
              "results": results, "failures": failures}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"QUALITY_DONE measured={len(results)} failed={len(failures)}", flush=True)


if __name__ == "__main__":
    main()
