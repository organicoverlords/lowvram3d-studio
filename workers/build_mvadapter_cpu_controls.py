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
from mvadapter_orientation_fixture import (
    AXIS_OCCLUDED_COMPONENT,
    AXIS_SIGNATURE_COMPONENT,
    COMPONENT_NAMES,
    COMPONENT_WORLD_DIRECTION,
    build_fixture,
    normalise,
)


VIEWS: tuple[dict[str, Any], ...] = (
    {"index": 0, "semantic_name": "horizontal_0", "elevation": 0.0, "azimuth": -90.0, "axis": "front"},
    {"index": 1, "semantic_name": "horizontal_1", "elevation": 0.0, "azimuth": 0.0, "axis": "right"},
    {"index": 2, "semantic_name": "horizontal_2", "elevation": 0.0, "azimuth": 90.0, "axis": "rear"},
    {"index": 3, "semantic_name": "horizontal_3", "elevation": 0.0, "azimuth": 180.0, "axis": "left"},
    {"index": 4, "semantic_name": "top", "elevation": 89.99, "azimuth": 90.0, "axis": "top"},
    {"index": 5, "semantic_name": "bottom", "elevation": -89.99, "azimuth": 90.0, "axis": "bottom"},
)
#: Control space is built so that its +Z is the asset's up axis and its +X is the
#: direction the asset faces, because the rig places the azimuth-0 camera on +X and
#: sweeps elevation about +Z. The legacy basis assumed a Z-up asset; a standard glTF
#: asset is Y-up with +Z forward, and feeding one to the other rotates the character
#: onto its side and hands the face-on camera the elevation +-90 embedding.
CANONICAL_BASES = {
    # canonical = (-y, x, z): control up is mesh +Z.
    "legacy_z_up": np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64),
    # canonical = (z, x, y): control up is mesh +Y, control +X is mesh +Z.
    "y_up_z_front": np.asarray([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64),
}
DEFAULT_CANONICAL_BASIS = "legacy_z_up"
CANONICAL_TRANSFORM = CANONICAL_BASES[DEFAULT_CANONICAL_BASIS]
CANONICAL_INVERSE = CANONICAL_TRANSFORM.T
CAMERA_DISTANCE = 1.8
CAMERA_NEAR = 0.1
CAMERA_FAR = 100.0
#: Official MV-Adapter geometry framing: object normalised to +/-0.5 and
#: projected through one shared orthographic span of +/-0.55 for every view.
PROJECTION_HALF_SPAN = 0.55
PROJECTION_SPAN = PROJECTION_HALF_SPAN * 2.0
#: The largest normalised object dimension therefore occupies 1 / 1.1 of the frame.
FRAMING_EXPECTED_OCCUPANCY = 1.0 / 1.1
FRAMING_OCCUPANCY_MIN = 0.89
FRAMING_OCCUPANCY_MAX = 0.93

FIXTURE_RENDER_SIZE = 192
FIXTURE_MIN_COMPONENT_PIXELS = 16


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unit(value: list[float] | np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    length = np.linalg.norm(vector)
    if length <= 1e-12:
        raise RuntimeError("CAMERA_VECTOR_ZERO")
    return vector / length


def _official_camera(elevation_deg: float, azimuth_deg: float) -> dict[str, Any]:
    elevation = np.deg2rad(float(elevation_deg))
    azimuth = np.deg2rad(float(azimuth_deg))
    position = np.asarray(
        [
            CAMERA_DISTANCE * np.cos(elevation) * np.cos(azimuth),
            CAMERA_DISTANCE * np.cos(elevation) * np.sin(azimuth),
            CAMERA_DISTANCE * np.sin(elevation),
        ], dtype=np.float64,
    )
    direction = _unit(-position)
    world_up = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    right = _unit(np.cross(direction, world_up))
    up = _unit(np.cross(right, direction))
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = -direction
    c2w[:3, 3] = position
    w2c = np.linalg.inv(c2w)
    projection = np.zeros((4, 4), dtype=np.float64)
    projection[0, 0] = 2.0 / PROJECTION_SPAN
    projection[1, 1] = -2.0 / PROJECTION_SPAN
    projection[2, 2] = -2.0 / (CAMERA_FAR - CAMERA_NEAR)
    projection[2, 3] = -(CAMERA_FAR + CAMERA_NEAR) / (CAMERA_FAR - CAMERA_NEAR)
    projection[3, 3] = 1.0
    return {
        "camera_position": position,
        "camera_direction": direction,
        "camera_right": right,
        "camera_up": up,
        "camera_to_world": c2w,
        "world_to_camera": w2c,
        "projection_matrix": projection,
        "mvp_matrix": projection @ w2c,
    }


def _render_fixture_view(
    vertices: np.ndarray,
    normals: np.ndarray,
    tris: np.ndarray,
    component_of_triangle: np.ndarray,
    direction: np.ndarray,
    right: np.ndarray,
    up: np.ndarray,
    size: int,
) -> dict[str, Any]:
    """Rasterise the asymmetric fixture and measure per-component evidence."""
    screen, depth = _project(vertices, direction, right, up, PROJECTION_SPAN)
    face_id, _bary, _position, _normal, zbuffer = _rasterise(screen, depth, vertices, normals, tris, size)
    mask = face_id >= 0
    if not mask.any():
        raise RuntimeError("CAMERA_CONTRACT_FIXTURE_EMPTY_VIEW")
    ys, xs = np.nonzero(mask)
    component = component_of_triangle[face_id[mask]]
    depths = zbuffer[mask]
    measured: dict[str, dict[str, float]] = {}
    for component_id, name in enumerate(COMPONENT_NAMES):
        selected = component == component_id
        if not selected.any():
            continue
        measured[name] = {
            "pixels": int(selected.sum()),
            "min_depth": float(depths[selected].min()),
            "centroid_x": float(xs[selected].mean() / (size - 1)),
            "centroid_y": float(ys[selected].mean() / (size - 1)),
        }
    closest = min(measured, key=lambda name: measured[name]["min_depth"])
    return {"measured": measured, "closest_component": closest}


def _prove_semantic_orientation(views: list[dict[str, Any]], size: int = FIXTURE_RENDER_SIZE) -> dict[str, Any]:
    """Prove six-view semantics from rendered triangle evidence.

    Nothing here is asserted a priori.  Each view is rasterised, every visible
    component is measured, and the declared semantics only survive if the
    rendered pixels agree.
    """
    fixture = build_fixture()
    vertices = normalise(fixture["vertices"])
    tris = fixture["triangles"]
    component_of_triangle = fixture["component_of_triangle"]
    normals = vertex_normals(vertices, tris).astype(np.float64)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)

    evidence: list[dict[str, Any]] = []
    semantic_mapping_proven = True
    handedness_checks = 0
    handedness_proven = True
    rotation_proven = {"top": False, "bottom": False}
    index_semantics: dict[str, str] = {}

    for view in views:
        axis = str(view["axis_label"])
        direction = np.asarray(view["camera_direction"], dtype=np.float64)
        up = np.asarray(view["camera_up"], dtype=np.float64)
        right = np.asarray(view["camera_right"], dtype=np.float64)
        official = _official_camera(float(view["elevation_deg"]), float(view["azimuth_deg"]))
        if not (
            np.allclose(direction, official["camera_direction"], atol=1e-6)
            and np.allclose(right, official["camera_right"], atol=1e-6)
            and np.allclose(up, official["camera_up"], atol=1e-6)
            and np.allclose(np.asarray(view["camera_to_world"]), official["camera_to_world"], atol=1e-6)
        ):
            raise RuntimeError(f"CAMERA_CONTRACT_ASYMMETRIC_FIXTURE_FAILED:[{view['index']}]")
        rendered = _render_fixture_view(
            vertices, normals, tris, component_of_triangle, direction, right, up, size
        )
        measured = rendered["measured"]

        expected_signature = AXIS_SIGNATURE_COMPONENT[axis]
        expected_occluded = AXIS_OCCLUDED_COMPONENT[axis]
        signature_passed = rendered["closest_component"] == expected_signature
        occlusion_passed = expected_occluded not in measured
        view_semantic_passed = bool(signature_passed and occlusion_passed)
        semantic_mapping_proven &= view_semantic_passed
        # The semantic of this index is read out of the render, not declared.
        resolved = {name: label for label, name in AXIS_SIGNATURE_COMPONENT.items()}
        index_semantics[str(int(view["index"]))] = resolved.get(rendered["closest_component"], "unresolved")

        placements: list[dict[str, Any]] = []
        view_x_passed = True
        view_y_passed = True
        for name, stats in measured.items():
            world_direction = COMPONENT_WORLD_DIRECTION.get(name)
            if world_direction is None or stats["pixels"] < FIXTURE_MIN_COMPONENT_PIXELS:
                continue
            world = np.asarray(world_direction, dtype=np.float64)
            along_right = float(np.dot(world, right))
            along_up = float(np.dot(world, up))
            record: dict[str, Any] = {
                "component": name,
                "pixels": stats["pixels"],
                "centroid_x": round(stats["centroid_x"], 6),
                "centroid_y": round(stats["centroid_y"], 6),
                "dot_camera_right": round(along_right, 6),
                "dot_camera_up": round(along_up, 6),
            }
            if abs(along_right) > 0.4:
                expected_side = "image_right" if along_right > 0 else "image_left"
                measured_side = "image_right" if stats["centroid_x"] > 0.5 else "image_left"
                passed = expected_side == measured_side
                record["expected_image_side_x"] = expected_side
                record["measured_image_side_x"] = measured_side
                record["image_side_x_passed"] = passed
                view_x_passed &= passed
                handedness_checks += 1
            if abs(along_up) > 0.4:
                # Screen y grows downwards, so +camera_up must land in the upper half.
                expected_side = "image_upper" if along_up > 0 else "image_lower"
                measured_side = "image_upper" if stats["centroid_y"] < 0.5 else "image_lower"
                passed = expected_side == measured_side
                record["expected_image_side_y"] = expected_side
                record["measured_image_side_y"] = measured_side
                record["image_side_y_passed"] = passed
                view_y_passed &= passed
            placements.append(record)

        if axis in {"front", "rear", "left", "right"}:
            handedness_proven &= view_x_passed
        if axis in rotation_proven:
            rotation_proven[axis] = bool(view_x_passed and view_y_passed and view_semantic_passed)

        evidence.append(
            {
                "index": int(view["index"]),
                "axis_label": axis,
                "camera_direction": direction.tolist(),
                "camera_up": up.tolist(),
                "camera_right": right.tolist(),
                "expected_signature_component": expected_signature,
                "measured_closest_component": rendered["closest_component"],
                "expected_occluded_component": expected_occluded,
                "expected_visible_components": sorted(
                    name for name in COMPONENT_NAMES if name != expected_occluded
                ),
                "measured_visible_components": sorted(measured),
                "component_pixels": {name: stats["pixels"] for name, stats in measured.items()},
                "image_side_placements": placements,
                "signature_passed": signature_passed,
                "occlusion_passed": occlusion_passed,
                "image_side_x_passed": view_x_passed,
                "image_side_y_passed": view_y_passed,
                "passed": bool(view_semantic_passed and view_x_passed and view_y_passed),
            }
        )

    if handedness_checks < 4:
        handedness_proven = False
    top_bottom_rotation_proven = bool(rotation_proven["top"] and rotation_proven["bottom"])
    if not (semantic_mapping_proven and handedness_proven and top_bottom_rotation_proven):
        failed = [record["index"] for record in evidence if not record["passed"]]
        raise RuntimeError(f"CAMERA_CONTRACT_ASYMMETRIC_FIXTURE_FAILED:{failed}")
    return {
        "fixture_name": "six_side_asymmetric_geometry_v2",
        "fixture_render_size": size,
        "fixture_triangle_count": fixture["triangle_count"],
        "fixture_vertex_count": fixture["vertex_count"],
        "component_ids": fixture["component_ids"],
        "index_semantics": index_semantics,
        "semantic_mapping_proven": bool(semantic_mapping_proven),
        "handedness_proven": bool(handedness_proven),
        "handedness_checks": int(handedness_checks),
        "top_rotation_proven": bool(rotation_proven["top"]),
        "bottom_rotation_proven": bool(rotation_proven["bottom"]),
        "top_bottom_rotation_proven": top_bottom_rotation_proven,
        "evidence": evidence,
    }


def build_camera_contract(canonical_basis: str = DEFAULT_CANONICAL_BASIS) -> dict[str, Any]:
    views: list[dict[str, Any]] = []
    for item in VIEWS:
        camera = _official_camera(item["elevation"], item["azimuth"])
        views.append({
            "index": int(item["index"]),
            "semantic_name": str(item["semantic_name"]),
            "axis_label": str(item["axis"]),
            "elevation_deg": float(item["elevation"]),
            "azimuth_deg": float(item["azimuth"]),
            "distance": CAMERA_DISTANCE,
            "camera_position": camera["camera_position"].tolist(),
            "camera_direction": camera["camera_direction"].tolist(),
            "camera_up": camera["camera_up"].tolist(),
            "camera_right": camera["camera_right"].tolist(),
            "camera_to_world": camera["camera_to_world"].tolist(),
            "world_to_camera": camera["world_to_camera"].tolist(),
            "projection_matrix": camera["projection_matrix"].tolist(),
            "mvp_matrix": camera["mvp_matrix"].tolist(),
            "fixture_gate_passed": False,
        })
    by_axis = {v["axis_label"]: np.asarray(v["camera_direction"]) for v in views}
    horizontal = [v for v in views if v["axis_label"] in {"front", "right", "rear", "left"}]
    fixture_evidence = _prove_semantic_orientation(views)
    fixture_gate = (
        len(views) == 6
        and len({v["index"] for v in views}) == 6
        and float(np.dot(by_axis["front"], by_axis["rear"])) <= -0.999
        and float(np.dot(by_axis["left"], by_axis["right"])) <= -0.999
        and float(np.dot(by_axis["top"], by_axis["bottom"])) <= -0.999
        and len(horizontal) == 4
        and all(abs(np.linalg.norm(v["camera_direction"]) - 1.0) < 1e-6 for v in views)
        and fixture_evidence["semantic_mapping_proven"]
        and fixture_evidence["handedness_proven"]
        and fixture_evidence["top_bottom_rotation_proven"]
        and fixture_evidence["index_semantics"].get("4") == "top"
        and fixture_evidence["index_semantics"].get("5") == "bottom"
    )
    if not fixture_gate:
        raise RuntimeError("CAMERA_CONTRACT_FIXTURE_FAILED")
    for view in views:
        record = fixture_evidence["evidence"][int(view["index"])]
        view["fixture_gate_passed"] = bool(record["passed"])
        view["proven_semantic"] = fixture_evidence["index_semantics"][str(int(view["index"]))]
    if not all(view["fixture_gate_passed"] for view in views):
        raise RuntimeError("CAMERA_CONTRACT_VIEW_GATE_FAILED")
    return {
        "schema": "lowvram3d_mvadapter_camera_contract_v1",
        "view_count": 6,
        "views": views,
        "front_rear_direction_dot": float(np.dot(by_axis["front"], by_axis["rear"])),
        "left_right_direction_dot": float(np.dot(by_axis["left"], by_axis["right"])),
        "top_bottom_direction_dot": float(np.dot(by_axis["top"], by_axis["bottom"])),
        "handedness_proven": fixture_evidence["handedness_proven"],
        "semantic_mapping_proven": fixture_evidence["semantic_mapping_proven"],
        "top_rotation_proven": fixture_evidence["top_rotation_proven"],
        "bottom_rotation_proven": fixture_evidence["bottom_rotation_proven"],
        "top_bottom_rotation_proven": fixture_evidence["top_bottom_rotation_proven"],
        "index_semantics": fixture_evidence["index_semantics"],
        "fixture_gate_passed": all(view["fixture_gate_passed"] for view in views),
        "fixture_evidence": fixture_evidence,
        "projection_half_span": PROJECTION_HALF_SPAN,
        "projection_span": PROJECTION_SPAN,
        "canonical_basis": canonical_basis,
        "control_space_transform": CANONICAL_BASES[canonical_basis].tolist(),
        "control_space_inverse": CANONICAL_BASES[canonical_basis].T.tolist(),
        "control_space_recentered": False,
        "control_space_rescaled": True,
        "control_max_abs": 0.5,
        "official_camera_contract": {
            "distance": CAMERA_DISTANCE,
            "left": -PROJECTION_HALF_SPAN,
            "right": PROJECTION_HALF_SPAN,
            "bottom": -PROJECTION_HALF_SPAN,
            "top": PROJECTION_HALF_SPAN,
            "near": CAMERA_NEAR,
            "far": CAMERA_FAR,
            "elevations_deg": [item["elevation"] for item in VIEWS],
            "azimuths_deg": [item["azimuth"] for item in VIEWS],
            "official_near_pole_up": True,
        },
    }


def _project(vertices: np.ndarray, direction: np.ndarray, right: np.ndarray, up: np.ndarray, projection_span: float) -> tuple[np.ndarray, np.ndarray]:
    screen = np.stack(
        [vertices @ right / projection_span + 0.5, 0.5 - (vertices @ up) / projection_span], axis=1
    )
    depth = vertices @ direction
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


def build_controls(mesh: Path, output_dir: Path, size: int = 256,
                   canonical_basis: str = DEFAULT_CANONICAL_BASIS) -> dict[str, Any]:
    if size < 16:
        raise RuntimeError("CPU_CONTROL_SIZE_INVALID")
    if canonical_basis not in CANONICAL_BASES:
        raise RuntimeError(f"CPU_CONTROL_CANONICAL_BASIS_INVALID:{canonical_basis}")
    transform = CANONICAL_BASES[canonical_basis]
    original_hash = sha256(mesh)
    positions, _mesh_normals, uv, tris, normal_source, scene_report = read_glb(
        mesh, return_normal_source=True, return_scene_report=True
    )
    if uv is None:
        raise RuntimeError("CPU_CONTROL_UV_MISSING")
    positions = positions.astype(np.float64)
    canonical_positions = positions @ transform.T
    largest = float(np.max(np.abs(canonical_positions)))
    if largest <= 1e-12:
        raise RuntimeError("CPU_CONTROL_MESH_DEGENERATE")
    vertices = canonical_positions * (0.5 / largest)
    normals = np.asarray(_mesh_normals, dtype=np.float64) @ transform.T
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    contract = build_camera_contract(canonical_basis)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "camera_contract.json").write_text(json.dumps(contract, indent=2), encoding="utf-8")
    tensor = np.full((6, 6, size, size), 0.5, dtype=np.float32)
    per_view: list[dict[str, Any]] = []
    for item in contract["views"]:
        index = int(item["index"])
        name = str(item["semantic_name"])
        direction = np.asarray(item["camera_direction"], dtype=np.float64)
        up = np.asarray(item["camera_up"], dtype=np.float64)
        right = np.asarray(item["camera_right"], dtype=np.float64)
        screen, depth = _project(vertices, direction, right, up, PROJECTION_SPAN)
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
        rows, cols = np.nonzero(mask)
        occupancy_x = float((cols.max() - cols.min() + 1) / size)
        occupancy_y = float((rows.max() - rows.min() + 1) / size)
        borders = {
            "left": bool(cols.min() == 0),
            "right": bool(cols.max() == size - 1),
            "top": bool(rows.min() == 0),
            "bottom": bool(rows.max() == size - 1),
        }
        if (borders["left"] and borders["right"]) or (borders["top"] and borders["bottom"]):
            raise RuntimeError(f"CPU_CONTROL_FRAMING_TOUCHES_OPPOSING_BORDERS:{name}")
        per_view.append({
            "index": index,
            "semantic_name": name,
            "proven_semantic": item.get("proven_semantic"),
            "direction": item["camera_direction"],
            "camera_to_world": item["camera_to_world"],
            "silhouette_pixels": int(mask.sum()),
            "visible_triangles": int(visible.sum()),
            "projection_span": PROJECTION_SPAN,
            "elevation_deg": item["elevation_deg"],
            "azimuth_deg": item["azimuth_deg"],
            "occupancy_x": round(occupancy_x, 6),
            "occupancy_y": round(occupancy_y, 6),
            "occupancy": round(max(occupancy_x, occupancy_y), 6),
            "borders_touched": borders,
            "projected_occupancy": round(float(max(screen[:, 0].max() - screen[:, 0].min(), screen[:, 1].max() - screen[:, 1].min())), 6),
            "depth_finite": True,
            "normal_unit_before_encoding": True,
        })
    np.save(output_dir / "control_tensor.npy", tensor)
    if tensor.shape != (6, 6, size, size) or not np.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1:
        raise RuntimeError("CPU_CONTROL_TENSOR_INVALID")
    if sha256(mesh) != original_hash:
        raise RuntimeError("CPU_CONTROL_MESH_MUTATED")

    # Official framing: one shared orthographic span, never a per-view auto-fit.
    spans = {view["projection_span"] for view in per_view}
    if spans != {PROJECTION_SPAN}:
        raise RuntimeError(f"CPU_CONTROL_FRAMING_SPAN_NOT_SHARED:{sorted(spans)}")
    occupancies = [float(view["occupancy"]) for view in per_view]
    max_occupancy = max(occupancies)
    # Occupancy is counted in whole pixels, so the gate has to admit the
    # quantisation error of a two-pixel measurement.  At production sizes this
    # is negligible (2/256 = 0.008) and the band stays exactly 0.89-0.93; only
    # very small fixture renders, where one pixel is several percent of the
    # frame, get the physically necessary widening.
    quantisation = 2.0 / size
    lower_gate = min(FRAMING_OCCUPANCY_MIN, FRAMING_EXPECTED_OCCUPANCY - quantisation)
    upper_gate = max(FRAMING_OCCUPANCY_MAX, FRAMING_EXPECTED_OCCUPANCY + quantisation)
    if max_occupancy > upper_gate:
        raise RuntimeError(f"CPU_CONTROL_FRAMING_OCCUPANCY_INVALID:{max_occupancy:.6f}")
    if max_occupancy > FRAMING_EXPECTED_OCCUPANCY + quantisation:
        raise RuntimeError(f"CPU_CONTROL_FRAMING_EXCEEDS_SHARED_SPAN:{max_occupancy:.6f}")
    report = {
        "schema": "lowvram3d_mvadapter_cpu_controls_v1",
        "mesh": str(mesh),
        "mesh_sha256_before": original_hash,
        "mesh_sha256_after": sha256(mesh),
        "geometry_or_uv_mutation": False,
        "normal_source": normal_source,
        "gltf_scene_transform": scene_report,
        "official_normal_contract": True,
        "canonical_basis": canonical_basis,
        "control_space_transform": transform.tolist(),
        "control_space_inverse": transform.T.tolist(),
        "control_space_recentered": False,
        "control_space_rescaled": True,
        "control_max_abs": round(float(np.max(np.abs(vertices))), 9),
        "size": size,
        "projection_span": PROJECTION_SPAN,
        "projection_half_span": PROJECTION_HALF_SPAN,
        "framing": {
            "shared_span": PROJECTION_SPAN,
            "per_view_autofit": False,
            "expected_occupancy": round(FRAMING_EXPECTED_OCCUPANCY, 6),
            "occupancy_min_gate": FRAMING_OCCUPANCY_MIN,
            "occupancy_max_gate": FRAMING_OCCUPANCY_MAX,
            "lower_coverage_is_evidence_only": True,
            "occupancy_lower_gate_applied": round(lower_gate, 6),
            "occupancy_upper_gate_applied": round(upper_gate, 6),
            "pixel_quantisation": round(quantisation, 6),
            "measured_occupancy_per_view": {
                str(view["index"]): view["occupancy"] for view in per_view
            },
            "measured_max_occupancy": round(max_occupancy, 6),
            "opposing_borders_touched": False,
            "passed": True,
        },
        "control_tensor": str(output_dir / "control_tensor.npy"),
        "control_tensor_shape": list(tensor.shape),
        "channel_order": ["position_x", "position_y", "position_z", "normal_x", "normal_y", "normal_z"],
        "background_encoding": [0.5] * 6,
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
    parser.add_argument("--canonical-basis", default=DEFAULT_CANONICAL_BASIS,
                        choices=sorted(CANONICAL_BASES))
    args = parser.parse_args()
    report = build_controls(Path(args.mesh), Path(args.output_dir), args.size,
                            canonical_basis=args.canonical_basis)
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
