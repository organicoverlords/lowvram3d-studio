"""Build CPU-owned normal/depth/alpha/triangle-ID conditions for four panda views.

The source view PNGs are already-rendered evidence from the immutable chart-separated
mesh. This worker only rasterizes the same mesh again to make explicit conditions
for the hybrid image-reference lane; it never edits the mesh or its UVs.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mesh_io import face_normals, read_glb
from build_projection_repair_bundle import project, rasterise_face_ids


VIEWS = {
    "front": np.array([0.0, -1.0, 0.0], dtype=np.float64),
    "left": np.array([-1.0, 0.0, 0.0], dtype=np.float64),
    "right": np.array([1.0, 0.0, 0.0], dtype=np.float64),
    "rear": np.array([0.0, 1.0, 0.0], dtype=np.float64),
}


def write_conditions(mesh: Path, rendered_dir: Path, output_dir: Path, size: int = 512) -> dict:
    positions, _mesh_normals, _uv, tris = read_glb(mesh)
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    ortho = float((verts.max(axis=0) - verts.min(axis=0)).max())
    if ortho <= 0 or size < 64:
        raise RuntimeError("CONDITION_BUNDLE_INVALID_MESH_BOUNDS")
    normals = face_normals(verts, tris).astype(np.float32)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, object] = {}
    for name, direction in VIEWS.items():
        source_name = "back" if name == "rear" else name
        source_path = rendered_dir / f"{source_name}.png"
        source = cv2.imread(str(source_path), cv2.IMREAD_UNCHANGED)
        if source is None:
            raise RuntimeError(f"CONDITION_SOURCE_MISSING:{source_path}")
        if source.shape[:2] != (size, size):
            source = cv2.resize(source, (size, size), interpolation=cv2.INTER_AREA)
        screen, depth_vertices = project(verts, direction, ortho)
        face_id, visible = rasterise_face_ids(screen, depth_vertices, tris, size)
        silhouette = face_id >= 0
        depth = np.zeros((size, size), np.float32)
        normal_rgb = np.zeros((size, size, 3), np.float32)
        if silhouette.any():
            ids = face_id[silhouette]
            depth[silhouette] = depth_vertices[tris[ids]].mean(axis=1)
            normal_rgb[silhouette] = normals[ids]
        dmin = float(depth[silhouette].min()) if silhouette.any() else 0.0
        dmax = float(depth[silhouette].max()) if silhouette.any() else 1.0
        depth_u8 = np.zeros((size, size), np.uint8)
        if silhouette.any() and dmax > dmin:
            depth_u8[silhouette] = np.clip(
                (depth[silhouette] - dmin) / (dmax - dmin) * 255.0, 0, 255
            ).astype(np.uint8)
        normal_u8 = np.zeros((size, size, 3), np.uint8)
        normal_u8[silhouette] = np.clip(
            (normal_rgb[silhouette] * 0.5 + 0.5) * 255.0, 0, 255
        ).astype(np.uint8)
        alpha = silhouette.astype(np.uint8) * 255
        view_dir = direction / np.linalg.norm(direction)
        camera_location = view_dir * (ortho * 3.0)
        transform = np.eye(4, dtype=np.float32)
        transform[:3, 3] = camera_location.astype(np.float32)
        cv2.imwrite(str(output_dir / f"{name}_source.png"), source)
        cv2.imwrite(str(output_dir / f"{name}_normal.png"), cv2.cvtColor(normal_u8, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(output_dir / f"{name}_depth.png"), depth_u8)
        cv2.imwrite(str(output_dir / f"{name}_mask.png"), alpha)
        np.save(output_dir / f"{name}_triangle_ids.npy", face_id)
        reports[name] = {
            "source": str(source_path),
            "direction": view_dir.tolist(),
            "camera_transform": transform.tolist(),
            "triangle_id": str(output_dir / f"{name}_triangle_ids.npy"),
            "visible_triangles": int(visible.sum()),
            "silhouette_pixels": int(np.count_nonzero(silhouette)),
            "depth_range": [dmin, dmax],
        }
    dot_front_rear = float(np.dot(VIEWS["front"], VIEWS["rear"]))
    report = {
        "schema": "lowvram3d_hybrid_condition_bundle_v1",
        "mesh": str(mesh),
        "mesh_triangles": int(len(tris)),
        "size": size,
        "view_order": list(VIEWS),
        "front_rear_direction_dot": dot_front_rear,
        "front_rear_gate_passed": dot_front_rear <= -0.999,
        "views": reports,
        "atlas_write": False,
        "geometry_or_uv_mutation": False,
    }
    (output_dir / "condition_bundle_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--rendered-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    report = write_conditions(
        Path(args.mesh), Path(args.rendered_dir), Path(args.output_dir), args.size
    )
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["front_rear_gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
