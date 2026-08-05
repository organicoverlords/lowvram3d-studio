"""Generic UV-chart and camera-registration diagnostics.

This utility deliberately contains no character names, paths, triangle tables,
or semantic assumptions.  A caller supplies a mesh, optional triangle mask,
and camera contract; the output maps selected triangles to UV charts and
camera-space projections so a face/feature registration issue can be diagnosed
without baking a model-specific exception into the production pipeline.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

try:  # Works both as ``python workers/tool.py`` and as a package import in tests.
    from .conservative_atlas import derive_uv_chart_ids
    from .mesh_io import read_glb
except ImportError:  # pragma: no cover - exercised by the script entry point.
    from conservative_atlas import derive_uv_chart_ids
    from mesh_io import read_glb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _camera_projection(points: np.ndarray, camera: dict) -> dict:
    right = np.asarray(camera["right"], dtype=np.float64)
    up = np.asarray(camera["up"], dtype=np.float64)
    origin = np.asarray(camera.get("origin", [0.0, 0.0, 0.0]), dtype=np.float64)
    if right.shape != (3,) or up.shape != (3,) or origin.shape != (3,):
        raise ValueError("camera vectors must have three components")
    width = float(camera.get("width", 1.0))
    height = float(camera.get("height", 1.0))
    if width <= 0 or height <= 0:
        raise ValueError("camera dimensions must be positive")
    delta = points - origin
    return {
        "x": (delta @ right).tolist(),
        "y": (delta @ up).tolist(),
        "width": width,
        "height": height,
    }


def build_report(mesh_path: Path, cameras_path: Path, output: Path,
                 triangle_mask_path: Path | None = None) -> dict:
    positions, normals, uv, triangles = read_glb(mesh_path)
    if uv is None:
        raise ValueError("mesh must contain TEXCOORD_0")
    chart_ids, chart_inventory = derive_uv_chart_ids(uv, triangles)
    mask = np.ones(len(triangles), dtype=bool)
    if triangle_mask_path is not None:
        mask = np.asarray(np.load(triangle_mask_path), dtype=bool)
        if mask.shape != (len(triangles),):
            raise ValueError("triangle mask must have one bool per triangle")

    tri_positions = positions[triangles]
    tri_centroids = tri_positions.mean(axis=1)
    tri_normals = normals[triangles].mean(axis=1)
    tri_normals /= np.maximum(np.linalg.norm(tri_normals, axis=1, keepdims=True), 1e-12)
    cameras = json.loads(cameras_path.read_text(encoding="utf-8"))
    if not isinstance(cameras, list):
        raise ValueError("camera contract must be a JSON list")

    selected = np.flatnonzero(mask)
    rows = []
    projections = {}
    for camera in cameras:
        name = str(camera["name"])
        projections[name] = _camera_projection(tri_centroids[selected], camera)
    for row_index, triangle_id in enumerate(selected.tolist()):
        row = {
            "triangle_id": int(triangle_id),
            "chart_id": int(chart_ids[triangle_id]),
            "centroid": tri_centroids[triangle_id].astype(float).tolist(),
            "normal": tri_normals[triangle_id].astype(float).tolist(),
            "uv": uv[triangles[triangle_id]].astype(float).tolist(),
            "camera_projections": {
                name: {
                    "x": float(values["x"][row_index]),
                    "y": float(values["y"][row_index]),
                }
                for name, values in projections.items()
            },
        }
        rows.append(row)
    report = {
        "schema": "surface_registration_forensics_v1",
        "mesh": str(mesh_path),
        "mesh_sha256": sha256(mesh_path),
        "triangle_count": int(len(triangles)),
        "selected_triangle_count": int(mask.sum()),
        "chart_count": int(chart_inventory["chart_count"]),
        "chart_inventory": chart_inventory,
        "camera_contract": cameras,
        "triangle_records": rows,
        "model_specific_exceptions": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=Path, required=True)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--triangle-mask", type=Path)
    args = parser.parse_args()
    print(json.dumps(build_report(args.mesh, args.cameras, args.output, args.triangle_mask), indent=2))


if __name__ == "__main__":
    main()
