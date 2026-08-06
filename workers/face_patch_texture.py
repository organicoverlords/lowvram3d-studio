"""Patch an authoritative source face onto an explicitly owned 3-D surface.

The baseline atlas is copied byte-for-byte. Only atlas texels whose exact owner
triangle belongs to the selected surface patch may change. A thin-plate spline
maps projected target-surface coordinates to authoritative source coordinates.
Geometry, indices and UVs are preserved by the existing clean texture binder.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .atlas_raster import rasterise
    from .face_surface_ownership import Camera
    from .fast_texture_projection import bind_texture, immutable_buffer_hashes
    from .mesh_io import read_glb
except ImportError:  # pragma: no cover
    from atlas_raster import rasterise
    from face_surface_ownership import Camera
    from fast_texture_projection import bind_texture, immutable_buffer_hashes
    from mesh_io import read_glb

EPS = 1e-9


@dataclass(frozen=True)
class ThinPlateSpline:
    control: np.ndarray
    weights: np.ndarray
    affine: np.ndarray


def _kernel(radius_squared: np.ndarray) -> np.ndarray:
    radius_squared = np.asarray(radius_squared, np.float64)
    out = np.zeros_like(radius_squared)
    valid = radius_squared > EPS
    out[valid] = radius_squared[valid] * np.log(radius_squared[valid])
    return out


def fit_tps(source_xy: np.ndarray, target_xy: np.ndarray, regularization: float = 1e-4) -> ThinPlateSpline:
    source = np.asarray(source_xy, np.float64)
    target = np.asarray(target_xy, np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2 or len(source) < 3:
        raise ValueError("TPS_CONTROL_SHAPE_INVALID")
    if len(np.unique(source, axis=0)) != len(source):
        raise ValueError("TPS_DUPLICATE_CONTROL_POINT")
    delta = source[:, None, :] - source[None, :, :]
    kernel = _kernel(np.einsum("ijk,ijk->ij", delta, delta))
    kernel.flat[::len(source) + 1] += float(regularization)
    polynomial = np.column_stack((np.ones(len(source)), source))
    system = np.block([[kernel, polynomial], [polynomial.T, np.zeros((3, 3), np.float64)]])
    rhs = np.vstack((target, np.zeros((3, 2), np.float64)))
    solution = np.linalg.solve(system, rhs)
    return ThinPlateSpline(source, solution[:len(source)], solution[len(source):])


def evaluate_tps(model: ThinPlateSpline, points_xy: np.ndarray) -> np.ndarray:
    points = np.asarray(points_xy, np.float64)
    delta = points[:, None, :] - model.control[None, :, :]
    kernel = _kernel(np.einsum("ijk,ijk->ij", delta, delta))
    polynomial = np.column_stack((np.ones(len(points)), points))
    return kernel @ model.weights + polynomial @ model.affine


def derive_foreground_alpha(image: np.ndarray, threshold: float = 18.0) -> np.ndarray:
    rgb = np.asarray(image, np.uint8)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0).astype(np.float32)
    background = np.median(border, axis=0)
    distance = np.linalg.norm(rgb.astype(np.float32) - background, axis=2)
    alpha = np.clip((distance - threshold) / max(threshold, 1.0), 0.0, 1.0)
    mask = (alpha > 0.1).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    alpha = np.maximum(alpha, cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 1.2))
    return np.clip(alpha, 0.0, 1.0)


def sample_premultiplied(image: np.ndarray, alpha: np.ndarray, xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(image, np.float32)
    alpha = np.asarray(alpha, np.float32)
    xy = np.asarray(xy, np.float64)
    height, width = alpha.shape
    x = xy[:, 0]
    y = xy[:, 1]
    valid = (x >= 0.0) & (x <= width - 1.0) & (y >= 0.0) & (y <= height - 1.0)
    x0 = np.floor(np.clip(x, 0, width - 1)).astype(np.int64)
    y0 = np.floor(np.clip(y, 0, height - 1)).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    fx = (np.clip(x, 0, width - 1) - x0).astype(np.float32)
    fy = (np.clip(y, 0, height - 1) - y0).astype(np.float32)
    weights = np.column_stack(((1-fx)*(1-fy), fx*(1-fy), (1-fx)*fy, fx*fy))
    alpha_values = np.column_stack((alpha[y0, x0], alpha[y0, x1], alpha[y1, x0], alpha[y1, x1]))
    premul = image * alpha[..., None]
    colour_values = np.stack((premul[y0, x0], premul[y0, x1], premul[y1, x0], premul[y1, x1]), axis=1)
    sampled_alpha = np.sum(weights * alpha_values, axis=1)
    sampled_premul = np.sum(weights[..., None] * colour_values, axis=1)
    sampled = sampled_premul / np.maximum(sampled_alpha[:, None], EPS)
    sampled[~valid] = 0.0
    sampled_alpha[~valid] = 0.0
    return sampled, sampled_alpha


def load_anchors(path: Path, positions: np.ndarray, triangles: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or len(payload) < 3:
        raise ValueError("FACE_ANCHORS_REQUIRE_AT_LEAST_THREE_RECORDS")
    target_points = []
    source_points = []
    records = []
    for record in payload:
        triangle_id = int(record["triangle_id"])
        bary = np.asarray(record["barycentric"], np.float64)
        source_xy = np.asarray(record["source_xy"], np.float64)
        if triangle_id < 0 or triangle_id >= len(triangles):
            raise ValueError("FACE_ANCHOR_TRIANGLE_OUT_OF_RANGE")
        if bary.shape != (3,) or np.any(bary < -1e-6) or abs(float(bary.sum()) - 1.0) > 1e-5:
            raise ValueError("FACE_ANCHOR_BARYCENTRIC_INVALID")
        if source_xy.shape != (2,):
            raise ValueError("FACE_ANCHOR_SOURCE_INVALID")
        point_3d = bary @ positions[triangles[triangle_id]]
        projected, depth = camera.project(point_3d[None, :])
        target_points.append(projected[0])
        source_points.append(source_xy)
        records.append({
            "name": str(record.get("name", f"anchor_{len(records)}")),
            "triangle_id": triangle_id,
            "barycentric": bary.tolist(),
            "point_3d": point_3d.tolist(),
            "target_xy": projected[0].tolist(),
            "target_depth": float(depth[0]),
            "source_xy": source_xy.tolist(),
        })
    return np.asarray(target_points), np.asarray(source_points), records


def build_face_patch_atlas(baseline_atlas: np.ndarray, source_image: np.ndarray,
                           source_alpha: np.ndarray, positions: np.ndarray, uv: np.ndarray,
                           triangles: np.ndarray, selected_triangles: np.ndarray,
                           camera: Camera, target_xy: np.ndarray, source_xy: np.ndarray,
                           minimum_alpha: float = 0.35,
                           tps_regularization: float = 1e-4) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    baseline = np.asarray(baseline_atlas, np.uint8)
    if baseline.ndim != 3 or baseline.shape[2] != 3 or baseline.shape[0] != baseline.shape[1]:
        raise ValueError("BASELINE_ATLAS_MUST_BE_SQUARE_RGB")
    size = baseline.shape[0]
    owner, barycentric_ab = rasterise(uv, triangles, size)
    if barycentric_ab.shape != (size, size, 2):
        raise ValueError("ATLAS_RASTER_BARYCENTRIC_CONTRACT_INVALID")
    selected_mask = np.zeros(len(triangles), bool)
    selected_ids = np.unique(np.asarray(selected_triangles, np.int64))
    if selected_ids.size and (selected_ids.min() < 0 or selected_ids.max() >= len(triangles)):
        raise ValueError("SELECTED_FACE_TRIANGLE_OUT_OF_RANGE")
    selected_mask[selected_ids] = True
    writable = (owner >= 0) & selected_mask[np.maximum(owner, 0)]
    rows, cols = np.nonzero(writable)
    triangle_ids = owner[writable].astype(np.int64)
    weights_ab = barycentric_ab[writable].astype(np.float64)
    weights = np.column_stack((1.0 - weights_ab[:, 0] - weights_ab[:, 1], weights_ab))
    if np.any(weights < -1e-5) or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5):
        raise RuntimeError("ATLAS_RASTER_BARYCENTRIC_RECONSTRUCTION_FAILED")
    points_3d = np.einsum("ij,ijk->ik", weights, positions[triangles[triangle_ids]])
    projected, depth = camera.project(points_3d)
    model = fit_tps(target_xy, source_xy, regularization=tps_regularization)
    source_sample_xy = evaluate_tps(model, projected)
    sampled, sampled_alpha = sample_premultiplied(source_image, source_alpha, source_sample_xy)
    accepted = sampled_alpha >= minimum_alpha
    out = baseline.copy()
    out[rows[accepted], cols[accepted]] = np.rint(np.clip(sampled[accepted], 0, 255)).astype(np.uint8)
    changed = np.any(out != baseline, axis=2)
    non_face = ~writable
    if np.any(out[non_face] != baseline[non_face]):
        raise RuntimeError("NON_FACE_ATLAS_BYTES_CHANGED")
    report = {
        "schema": "face_patch_texture_v2",
        "atlas_size": size,
        "atlas_raster_barycentric_components": 2,
        "selected_triangle_count": int(len(selected_ids)),
        "writable_face_texels": int(writable.sum()),
        "accepted_face_texels": int(accepted.sum()),
        "rejected_low_alpha_texels": int((~accepted).sum()),
        "changed_face_texels": int(changed.sum()),
        "non_face_atlas_pixels_changed": 0,
        "projected_depth_min": float(np.nanmin(depth)) if len(depth) else None,
        "projected_depth_max": float(np.nanmax(depth)) if len(depth) else None,
        "tps_control_count": int(len(target_xy)),
        "tps_regularization": float(tps_regularization),
        "minimum_alpha": float(minimum_alpha),
    }
    return out, report, writable


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glb", type=Path, required=True)
    parser.add_argument("--baseline-atlas", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--source-alpha", type=Path)
    parser.add_argument("--camera", type=Path, required=True)
    parser.add_argument("--selected-triangles", type=Path, required=True)
    parser.add_argument("--anchors", type=Path, required=True)
    parser.add_argument("--output-atlas", type=Path, required=True)
    parser.add_argument("--output-glb", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--minimum-alpha", type=float, default=0.35)
    parser.add_argument("--tps-regularization", type=float, default=1e-4)
    args = parser.parse_args()

    positions, normals, uv, triangles = read_glb(args.baseline_glb)
    if uv is None:
        raise RuntimeError("FACE_PATCH_REQUIRES_UV")
    atlas_bgr = cv2.imread(str(args.baseline_atlas), cv2.IMREAD_COLOR)
    source_bgr = cv2.imread(str(args.source_image), cv2.IMREAD_COLOR)
    if atlas_bgr is None or source_bgr is None:
        raise RuntimeError("FACE_PATCH_IMAGE_MISSING")
    atlas = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    source = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    if args.source_alpha:
        alpha_raw = cv2.imread(str(args.source_alpha), cv2.IMREAD_GRAYSCALE)
        if alpha_raw is None:
            raise RuntimeError("FACE_PATCH_ALPHA_MISSING")
        alpha = alpha_raw.astype(np.float32) / 255.0
    else:
        alpha = derive_foreground_alpha(source)
    payload = json.loads(args.camera.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1:
            raise ValueError("FACE_PATCH_CAMERA_MUST_CONTAIN_ONE_CAMERA")
        payload = payload[0]
    camera = Camera.from_dict(payload)
    target_xy, source_xy, anchor_records = load_anchors(args.anchors, positions, triangles, camera)
    selected = np.asarray(np.load(args.selected_triangles), np.int64)
    before_hashes = immutable_buffer_hashes(args.baseline_glb)
    result, report, writable = build_face_patch_atlas(
        atlas, source, alpha, positions, uv, triangles, selected, camera,
        target_xy, source_xy, minimum_alpha=args.minimum_alpha,
        tps_regularization=args.tps_regularization,
    )
    args.output_atlas.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(result, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("FACE_PATCH_PNG_ENCODE_FAILED")
    args.output_atlas.write_bytes(encoded.tobytes())
    textured_triangles = np.ones(len(triangles), bool)
    bind_texture(args.baseline_glb, args.output_glb, encoded.tobytes(), textured_triangles=textured_triangles)
    after_hashes = immutable_buffer_hashes(args.output_glb)
    geometry_preserved = before_hashes == after_hashes
    report.update({
        "classification": "CANDIDATE_REQUIRES_VISUAL_REVIEW",
        "baseline_glb": str(args.baseline_glb),
        "baseline_glb_sha256": sha256(args.baseline_glb),
        "baseline_atlas": str(args.baseline_atlas),
        "baseline_atlas_sha256": sha256(args.baseline_atlas),
        "source_image": str(args.source_image),
        "source_image_sha256": sha256(args.source_image),
        "output_atlas": str(args.output_atlas),
        "output_atlas_sha256": sha256(args.output_atlas),
        "output_glb": str(args.output_glb),
        "output_glb_sha256": sha256(args.output_glb),
        "anchors": anchor_records,
        "immutable_hashes_before": before_hashes,
        "immutable_hashes_after": after_hashes,
        "geometry_uv_index_preserved": geometry_preserved,
        "promotion_authorized": False,
    })
    if not geometry_preserved:
        raise RuntimeError("FACE_PATCH_GEOMETRY_OR_UV_CHANGED")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
