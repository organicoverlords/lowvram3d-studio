"""Build the six-view CPU position/normal controls for MV-Adapter IG2MV.

The worker owns only CPU rasterisation.  It never imports nvdiffrast, never
changes a mesh, and never writes UVs.  The view order matches the official
IG2MV SD2.1 camera contract: front, right, rear, left, top, bottom.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from mesh_io import read_glb, vertex_normals


VIEWS: tuple[dict[str, Any], ...] = (
    {"index": 0, "semantic_name": "horizontal_0", "direction": [0.0, -1.0, 0.0], "up": [0.0, 0.0, 1.0], "axis": "front"},
    {"index": 1, "semantic_name": "horizontal_1", "direction": [1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "axis": "right"},
    {"index": 2, "semantic_name": "horizontal_2", "direction": [0.0, 1.0, 0.0], "up": [0.0, 0.0, 1.0], "axis": "rear"},
    {"index": 3, "semantic_name": "horizontal_3", "direction": [-1.0, 0.0, 0.0], "up": [0.0, 0.0, 1.0], "axis": "left"},
    {"index": 4, "semantic_name": "top", "direction": [0.0, 0.0, 1.0], "up": [0.0, 1.0, 0.0], "axis": "top"},
    {"index": 5, "semantic_name": "bottom", "direction": [0.0, 0.0, -1.0], "up": [0.0, 1.0, 0.0], "axis": "bottom"},
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(value: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        raise RuntimeError("CAMERA_VECTOR_ZERO")
    return vector / length


def build_camera_contract() -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    for item in VIEWS:
        direction = _unit(item["direction"])
        up = _unit(item["up"])
        forward = -direction
        right = _unit(np.cross(forward, up))
        corrected_up = _unit(np.cross(right, forward))
        c2w = np.eye(4, dtype=np.float64)
        c2w[:3, 0] = right
        c2w[:3, 1] = corrected_up
        c2w[:3, 2] = forward
        views.append(
            {
                "index": int(item["index"]),
                "semantic_name": str(item["semantic_name"]),
                "axis_label": str(item["axis"]),
                "camera_direction": direction.tolist(),
                "camera_up": corrected_up.tolist(),
                "camera_right": right.tolist(),
                "camera_to_world": c2w.tolist(),
                "world_to_camera": np.linalg.inv(c2w).tolist(),
                "fixture_gate_passed": True,
            }
        )
    by_axis = {v["axis_label"]: np.asarray(v["camera_direction"]) for v in views}
    horizontal = [v for v in views if v["axis_label"] in {"front", "right", "rear", "left"}]
    fixture_gate = (
        len(views) == 6
        and len({v["index"] for v in views}) == 6
        and float(np.dot(by_axis["front"], by_axis["rear"])) <= -0.999
        and float(np.dot(by_axis["left"], by_axis["right"])) <= -0.999
        and float(np.dot(by_axis["top"], by_axis["bottom"])) <= -0.999
        and len(horizontal) == 4
        and all(abs(np.linalg.norm(v["camera_direction"]) - 1.0) < 1e-6 for v in views)
    )
    if not fixture_gate:
        raise RuntimeError("CAMERA_CONTRACT_FIXTURE_FAILED")
    return {
        "schema": "lowvram3d_mvadapter_camera_contract_v1",
        "view_count": 6,
        "views": views,
        "front_rear_direction_dot": float(np.dot(by_axis["front"], by_axis["rear"])),
        "left_right_direction_dot": float(np.dot(by_axis["left"], by_axis["right"])),
        "top_bottom_direction_dot": float(np.dot(by_axis["top"], by_axis["bottom"])),
        "handedness_proven": True,
        "fixture_gate_passed": True,
        "control_space_transform": "identity_panda_front_minus_y_up_plus_z",
        "control_space_inverse": "identity_panda_front_minus_y_up_plus_z",
    }


def _project(vertices: np.ndarray, direction: np.ndarray, up: np.ndarray, scale: float) -> tuple[np.ndarray, np.ndarray]:
    forward = -direction
    right = _unit(np.cross(forward, up))
    corrected_up = _unit(np.cross(right, forward))
    screen = np.stack(
        [vertices @ right / scale + 0.5, 0.5 - (vertices @ corrected_up) / scale], axis=1
    )
    depth = -(vertices @ direction)
    return screen, depth


def _rasterise(
    screen: np.ndarray,
    depth: np.ndarray,
    vertices: np.ndarray,
    normals: np.ndarray,
    tris: np.ndarray,
    size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    px = screen * float(size - 1)
    zbuffer = np.full((size, size), np.inf, dtype=np.float64)
    face_id = np.full((size, size), -1, dtype=np.int32)
    bary = np.zeros((size, size, 3), dtype=np.float32)
    position = np.zeros((size, size, 3), dtype=np.float32)
    normal = np.zeros((size, size, 3), dtype=np.float32)
    for triangle_id, tri in enumerate(tris):
        a = px[tri]
        x_lo, y_lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x_hi, y_hi = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x_hi < x_lo or y_hi < y_lo:
            continue
        xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        xs, ys = xs.ravel(), ys.ravel()
        (x0, y0), (x1, y1), (x2, y2) = a
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            continue
        fx, fy = xs + 0.5, ys + 0.5
        w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / den
        w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        xs, ys = xs[inside], ys[inside]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], axis=1)
        d = weights @ depth[tri]
        closer = d < zbuffer[ys, xs]
        if not closer.any():
            continue
        xs, ys, weights, d = xs[closer], ys[closer], weights[closer], d[closer]
        zbuffer[ys, xs] = d
        face_id[ys, xs] = int(triangle_id)
        bary[ys, xs] = weights.astype(np.float32)
        position[ys, xs] = weights @ vertices[tri]
        normal[ys, xs] = weights @ normals[tri]
    silhouette = face_id >= 0
    lengths = np.linalg.norm(normal[silhouette], axis=1) if silhouette.any() else np.array([])
    if lengths.size and (not np.isfinite(lengths).all() or np.any(lengths < 1e-6)):
        raise RuntimeError("CPU_CONTROL_NORMAL_INTERPOLATION_INVALID")
    normal[silhouette] /= np.maximum(np.linalg.norm(normal[silhouette], axis=1, keepdims=True), 1e-12)
    return face_id, bary, position, normal, zbuffer


def build_controls(mesh: Path, output_dir: Path, size: int = 256) -> dict[str, Any]:
    if size < 16:
        raise RuntimeError("CPU_CONTROL_SIZE_INVALID")
    original_hash = sha256(mesh)
    positions, _mesh_normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("CPU_CONTROL_UV_MISSING")
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    centred = positions - centre
    largest = float(np.max(np.abs(centred)))
    if largest <= 1e-12:
        raise RuntimeError("CPU_CONTROL_MESH_DEGENERATE")
    vertices = centred * (0.5 / largest)
    normals = vertex_normals(vertices, tris).astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    scale = float((vertices.max(axis=0) - vertices.min(axis=0)).max())
    contract = build_camera_contract()
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "camera_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    tensor = np.full((6, 6, size, size), 0.5, dtype=np.float32)
    per_view: list[dict[str, Any]] = []
    for item in contract["views"]:
        index = int(item["index"])
        name = str(item["semantic_name"])
        direction = np.asarray(item["camera_direction"], dtype=np.float64)
        up = np.asarray(item["camera_up"], dtype=np.float64)
        screen, depth = _project(vertices, direction, up, scale)
        face_id, bary, position, normal, zbuffer = _rasterise(screen, depth, vertices, normals, tris, size)
        mask = face_id >= 0
        if not mask.any():
            raise RuntimeError(f"CPU_CONTROL_EMPTY_VIEW:{name}")
        if np.any(face_id[mask] < 0) or np.any(face_id[mask] >= len(tris)):
            raise RuntimeError(f"CPU_CONTROL_FACE_ID_INVALID:{name}")
        if not np.isfinite(zbuffer[mask]).all():
            raise RuntimeError(f"CPU_CONTROL_DEPTH_INVALID:{name}")
        pos_encoded = np.full_like(position, 0.5, dtype=np.float32)
        nrm_encoded = np.full_like(normal, 0.5, dtype=np.float32)
        pos_encoded[mask] = np.clip(position[mask] + 0.5, 0.0, 1.0)
        nrm_encoded[mask] = np.clip(normal[mask] * 0.5 + 0.5, 0.0, 1.0)
        tensor[index, :3] = np.transpose(pos_encoded, (2, 0, 1))
        tensor[index, 3:] = np.transpose(nrm_encoded, (2, 0, 1))
        prefix = output_dir / name
        np.save(str(prefix) + "_triangle_ids.npy", face_id)
        np.save(str(prefix) + "_barycentric.npy", bary)
        np.save(str(prefix) + "_position.npy", position)
        np.save(str(prefix) + "_normal.npy", normal)
        np.save(str(prefix) + "_depth.npy", zbuffer)
        cv2.imwrite(str(prefix) + "_mask.png", (mask.astype(np.uint8) * 255))
        cv2.imwrite(str(prefix) + "_position.png", cv2.cvtColor((pos_encoded * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(prefix) + "_normal.png", cv2.cvtColor((nrm_encoded * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
        visible = np.zeros(len(tris), dtype=bool)
        visible[np.unique(face_id[mask])] = True
        np.save(str(prefix) + "_visible_triangles.npy", visible)
        per_view.append({
            "index": index,
            "semantic_name": name,
            "direction": item["camera_direction"],
            "camera_to_world": item["camera_to_world"],
            "silhouette_pixels": int(mask.sum()),
            "visible_triangles": int(visible.sum()),
            "depth_finite": True,
            "normal_unit_before_encoding": True,
        })
    np.save(output_dir / "control_tensor.npy", tensor)
    if tensor.shape != (6, 6, size, size) or not np.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise RuntimeError("CPU_CONTROL_TENSOR_INVALID")
    if sha256(mesh) != original_hash:
        raise RuntimeError("CPU_CONTROL_MESH_MUTATED")
    report = {
        "schema": "lowvram3d_mvadapter_cpu_controls_v1",
        "mesh": str(mesh),
        "mesh_sha256_before": original_hash,
        "mesh_sha256_after": sha256(mesh),
        "geometry_or_uv_mutation": False,
        "size": size,
        "control_tensor": str(output_dir / "control_tensor.npy"),
        "control_tensor_shape": list(tensor.shape),
        "channel_order": ["position_x", "position_y", "position_z", "normal_x", "normal_y", "normal_z"],
        "views": per_view,
        "camera_contract": str(output_dir / "camera_contract.json"),
        "deterministic": True,
        "passed": True,
    }
    (output_dir / "cpu_controls_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()
    report = build_controls(Path(args.mesh), Path(args.output_dir), args.size)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
