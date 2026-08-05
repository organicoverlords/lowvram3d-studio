"""CPU topology and silhouette metrics for the bounded ship geometry benchmark."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from mesh_io import read_glb, triangle_components

WELD = 4e-4
EXTREME_ASPECT = 50.0


def topology(positions: np.ndarray, triangles: np.ndarray) -> dict:
    labels, welded = triangle_components(positions, triangles, WELD)
    sizes = np.bincount(labels)
    edges = np.sort(np.concatenate((
        welded[triangles][:, [0, 1]],
        welded[triangles][:, [1, 2]],
        welded[triangles][:, [2, 0]],
    )), axis=1)
    unique, counts = np.unique(edges, axis=0, return_counts=True)
    return {
        "components": int(len(sizes)),
        "component_triangle_counts": sorted((int(v) for v in sizes), reverse=True)[:32],
        "detached_islands": int(max(0, len(sizes) - 1)),
        "boundary_edges": int(np.count_nonzero(counts == 1)),
        "non_manifold_edges": int(np.count_nonzero(counts > 2)),
        "welded_unique_edges": int(len(unique)),
    }


def measure(path: Path, view_dir: Path | None) -> dict:
    positions, _normals, _uv, triangles = read_glb(path)
    positions = np.asarray(positions, dtype=np.float64)
    triangles = np.asarray(triangles, dtype=np.int64)
    a, b, c = positions[triangles[:, 0]], positions[triangles[:, 1]], positions[triangles[:, 2]]
    lengths = np.stack((np.linalg.norm(b - a, axis=1), np.linalg.norm(c - b, axis=1), np.linalg.norm(a - c, axis=1)), axis=1)
    areas = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    repeated = (triangles[:, 0] == triangles[:, 1]) | (triangles[:, 1] == triangles[:, 2]) | (triangles[:, 0] == triangles[:, 2])
    aspect = lengths.max(axis=1) / np.maximum(lengths.min(axis=1), 1e-12)
    low, high = positions.min(axis=0), positions.max(axis=0)
    record = {
        "glb": str(path),
        "glb_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "vertices": int(len(positions)),
        "triangles": int(len(triangles)),
        "bounds_min": low.tolist(),
        "bounds_max": high.tolist(),
        "extent": (high - low).tolist(),
        "degenerate_triangles": int(np.count_nonzero(repeated | (areas <= 1e-12))),
        "duplicate_index_triangles": int(np.count_nonzero(repeated)),
        "extreme_aspect_threshold": EXTREME_ASPECT,
        "extreme_aspect_triangles": int(np.count_nonzero(aspect >= EXTREME_ASPECT)),
        "max_triangle_aspect": float(np.max(aspect)) if len(aspect) else 0.0,
        "topology": topology(positions, triangles),
    }
    if view_dir and view_dir.is_dir():
        silhouette = {}
        for image in sorted(view_dir.glob("*.png")):
            with Image.open(image) as opened:
                rgba = opened.convert("RGBA")
                alpha = np.asarray(rgba)[..., 3]
                silhouette[image.stem] = {
                    "dimensions": [int(rgba.width), int(rgba.height)],
                    "occupied_pixels": int(np.count_nonzero(alpha > 8)),
                    "occupancy_fraction": float(np.mean(alpha > 8)),
                }
        record["silhouette_occupancy"] = silhouette
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", action="append", required=True)
    parser.add_argument("--view-dir", action="append", default=[])
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    views = [Path(v) for v in args.view_dir]
    results = [measure(Path(mesh), views[index] if index < len(views) else None) for index, mesh in enumerate(args.mesh)]
    payload = {"schema": "ship_candidate_metrics_v1", "results": results}
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    for result in results:
        print(f"SHIP_METRICS {Path(result['glb']).name} tris={result['triangles']} components={result['topology']['components']} degenerate={result['degenerate_triangles']} extreme={result['extreme_aspect_triangles']}", flush=True)


if __name__ == "__main__":
    main()
