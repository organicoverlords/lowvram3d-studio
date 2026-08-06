"""End-to-end face-surface candidate builder using existing local evidence.

The worker derives the source camera from the foreground silhouette, restricts
raycasts to a bounded upper-central face region, selects a connected depth
surface, derives exact triangle/barycentric anchors from the selected layer,
and changes only texels owned by that surface in a copy of the baseline atlas.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .face_patch_texture import build_face_patch_atlas, derive_foreground_alpha
    from .face_surface_ownership import Camera, score_surface_patches, select_face_patch, trace_mask_layers
    from .fast_texture_projection import bind_texture, fit_camera, immutable_buffer_hashes
    from .conservative_atlas import derive_uv_chart_ids
    from .mesh_io import read_glb
except ImportError:  # pragma: no cover
    from face_patch_texture import build_face_patch_atlas, derive_foreground_alpha
    from face_surface_ownership import Camera, score_surface_patches, select_face_patch, trace_mask_layers
    from fast_texture_projection import bind_texture, fit_camera, immutable_buffer_hashes
    from conservative_atlas import derive_uv_chart_ids
    from mesh_io import read_glb

EPS = 1e-9


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def camera_from_projection_fit(fit: dict[str, Any], positions: np.ndarray,
                               image_width: int, image_height: int) -> Camera:
    matrix = np.asarray(fit["matrix"], np.float64)
    centre = np.asarray(fit["centre"], np.float64)
    offset = np.asarray(fit["offset"], np.float64)
    scale = float(fit["scale"])
    if matrix.shape != (3, 3) or centre.shape != (3,) or offset.shape != (2,) or scale <= 0:
        raise ValueError("FACE_CAMERA_FIT_INVALID")
    right = matrix[0] / max(float(np.linalg.norm(matrix[0])), EPS)
    source_y_axis = matrix[1] / max(float(np.linalg.norm(matrix[1])), EPS)
    forward = matrix[2] / max(float(np.linalg.norm(matrix[2])), EPS)
    up = -source_y_axis
    x_shift = (image_width * 0.5 - 0.5 - offset[0]) / scale
    y_shift = (image_height * 0.5 - 0.5 - offset[1]) / scale
    span = np.ptp(np.asarray(positions, np.float64), axis=0)
    depth_margin = max(float(np.linalg.norm(span)), 1.0) * 2.0
    origin = centre + x_shift * right + y_shift * source_y_axis - depth_margin * forward
    return Camera(
        origin=origin,
        right=right,
        up=up,
        forward=forward,
        width=int(image_width),
        height=int(image_height),
        projection="orthographic",
        ortho_width=float(image_width / scale),
        ortho_height=float(image_height / scale),
    )


def camera_to_dict(camera: Camera, fit: dict[str, Any]) -> dict[str, Any]:
    return {
        "origin": camera.origin.tolist(),
        "right": camera.right.tolist(),
        "up": camera.up.tolist(),
        "forward": camera.forward.tolist(),
        "width": camera.width,
        "height": camera.height,
        "projection": camera.projection,
        "ortho_width": camera.ortho_width,
        "ortho_height": camera.ortho_height,
        "fit": {
            key: (value.tolist() if isinstance(value, np.ndarray) else value)
            for key, value in fit.items()
        },
    }


def derive_face_mask(foreground: np.ndarray, *, top_fraction: float = 0.44,
                     central_fraction: float = 0.62) -> np.ndarray:
    foreground = np.asarray(foreground, bool)
    yy, xx = np.nonzero(foreground)
    if not len(xx):
        raise RuntimeError("FACE_FOREGROUND_EMPTY")
    x0, x1 = int(xx.min()), int(xx.max()) + 1
    y0, y1 = int(yy.min()), int(yy.max()) + 1
    width = x1 - x0
    height = y1 - y0
    roi_y1 = y0 + max(1, int(round(height * top_fraction)))
    half = max(1, int(round(width * central_fraction * 0.5)))
    centre_x = int(round((x0 + x1 - 1) * 0.5))
    roi_x0 = max(x0, centre_x - half)
    roi_x1 = min(x1, centre_x + half + 1)
    mask = np.zeros_like(foreground)
    mask[y0:roi_y1, roi_x0:roi_x1] = foreground[y0:roi_y1, roi_x0:roi_x1]
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel) > 0
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count > 1:
        central = (centre_x, int(round(y0 + (roi_y1 - y0) * 0.55)))
        label = int(labels[np.clip(central[1], 0, mask.shape[0]-1), np.clip(central[0], 0, mask.shape[1]-1)])
        if label == 0:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        mask = labels == label
    if int(mask.sum()) < 64:
        raise RuntimeError("FACE_MASK_TOO_SMALL")
    return mask


def _snap_to_mask(point: np.ndarray, mask_xy: np.ndarray) -> np.ndarray:
    distance = np.sum((mask_xy - point[None, :]) ** 2, axis=1)
    return mask_xy[int(np.argmin(distance))]


def landmarks_from_face_mask(mask: np.ndarray) -> list[dict[str, Any]]:
    yy, xx = np.nonzero(np.asarray(mask, bool))
    if not len(xx):
        raise RuntimeError("FACE_MASK_EMPTY")
    x0, x1 = float(xx.min()), float(xx.max())
    y0, y1 = float(yy.min()), float(yy.max())
    width = max(x1 - x0, 1.0)
    height = max(y1 - y0, 1.0)
    mask_xy = np.column_stack((xx, yy)).astype(np.float64)
    template = [
        ("left_eye", 0.34, 0.34),
        ("right_eye", 0.66, 0.34),
        ("nose", 0.50, 0.53),
        ("muzzle_left", 0.40, 0.64),
        ("muzzle_right", 0.60, 0.64),
        ("chin", 0.50, 0.76),
        ("forehead", 0.50, 0.18),
    ]
    records = []
    for name, u, v in template:
        target = np.asarray((x0 + u * width, y0 + v * height), np.float64)
        snapped = _snap_to_mask(target, mask_xy)
        records.append({"name": name, "source_xy": snapped.tolist(), "template_xy": target.tolist()})
    return records


def anchors_from_layers(layers: dict[str, np.ndarray], selected_triangles: np.ndarray,
                        landmarks: list[dict[str, Any]], *, neighbour_rays: int = 32) -> list[dict[str, Any]]:
    selected = set(np.asarray(selected_triangles, np.int64).tolist())
    pixels = np.asarray(layers["pixels_xy"], np.float64)
    offsets = np.asarray(layers["offsets"], np.int64)
    triangle_ids = np.asarray(layers["triangle_ids"], np.int64)
    barycentric = np.asarray(layers["barycentric"], np.float64)
    depth = np.asarray(layers["depth"], np.float64)
    anchors = []
    for landmark in landmarks:
        source_xy = np.asarray(landmark["source_xy"], np.float64)
        order = np.argsort(np.sum((pixels - source_xy[None, :]) ** 2, axis=1), kind="mergesort")
        chosen = None
        for ray_index in order[:max(1, neighbour_rays)]:
            start, end = int(offsets[ray_index]), int(offsets[ray_index + 1])
            candidates = [index for index in range(start, end) if int(triangle_ids[index]) in selected]
            if not candidates:
                continue
            hit_index = min(candidates, key=lambda index: (float(depth[index]), int(triangle_ids[index])))
            chosen = {
                "name": str(landmark["name"]),
                "source_xy": source_xy.tolist(),
                "sampled_ray_xy": pixels[ray_index].tolist(),
                "triangle_id": int(triangle_ids[hit_index]),
                "barycentric": barycentric[hit_index].tolist(),
                "depth": float(depth[hit_index]),
            }
            break
        if chosen is None:
            raise RuntimeError(f"FACE_ANCHOR_NOT_FOUND:{landmark['name']}")
        anchors.append(chosen)
    if len({anchor["triangle_id"] for anchor in anchors}) < 3:
        raise RuntimeError("FACE_ANCHORS_INSUFFICIENT_SURFACE_SPREAD")
    return anchors


def build_candidate(baseline_glb: Path, baseline_atlas: Path, source_image: Path,
                    output_dir: Path, *, ray_stride: int = 4,
                    minimum_alpha: float = 0.35) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    positions, normals, uv, triangles = read_glb(baseline_glb)
    if uv is None:
        raise RuntimeError("FACE_CANDIDATE_REQUIRES_UV")
    source_bgr = cv2.imread(str(source_image), cv2.IMREAD_COLOR)
    atlas_bgr = cv2.imread(str(baseline_atlas), cv2.IMREAD_COLOR)
    if source_bgr is None or atlas_bgr is None:
        raise RuntimeError("FACE_CANDIDATE_IMAGE_MISSING")
    source_rgb = cv2.cvtColor(source_bgr, cv2.COLOR_BGR2RGB)
    baseline_rgb = cv2.cvtColor(atlas_bgr, cv2.COLOR_BGR2RGB)
    source_alpha = derive_foreground_alpha(source_rgb)
    foreground = source_alpha >= 0.5
    camera_fit = fit_camera(np.asarray(positions, np.float64), np.asarray(triangles, np.int64), foreground)
    camera = camera_from_projection_fit(camera_fit, positions, source_rgb.shape[1], source_rgb.shape[0])
    camera_payload = camera_to_dict(camera, camera_fit)
    (output_dir / "camera_contract.json").write_text(json.dumps(camera_payload, indent=2), encoding="utf-8")
    face_mask = derive_face_mask(foreground)
    np.save(output_dir / "face_mask.npy", face_mask)
    cv2.imwrite(str(output_dir / "face_mask.png"), face_mask.astype(np.uint8) * 255)
    landmarks = landmarks_from_face_mask(face_mask)
    (output_dir / "source_landmarks.json").write_text(json.dumps(landmarks, indent=2), encoding="utf-8")
    chart_ids, chart_inventory = derive_uv_chart_ids(uv, triangles)
    layers = trace_mask_layers(positions, normals, triangles, camera, face_mask,
                               stride=ray_stride, max_hits=24, leaf_size=16)
    np.savez_compressed(output_dir / "all_ray_hits.npz", **layers)
    landmark_xy = np.asarray([record["source_xy"] for record in landmarks], np.float64)
    records = score_surface_patches(layers, triangles, positions, normals, chart_ids, camera, landmark_xy,
                                    max_rank=8, minimum_facing=-0.15)
    selected = select_face_patch(records, minimum_ray_coverage=0.02,
                                 maximum_side_wrap=0.48, minimum_landmark_support=0.25)
    selected_ids = np.asarray(selected["triangle_ids"], np.int64)
    np.save(output_dir / "selected_face_triangles.npy", selected_ids)
    anchors = anchors_from_layers(layers, selected_ids, landmarks)
    (output_dir / "auto_target_anchors.json").write_text(json.dumps(anchors, indent=2), encoding="utf-8")
    target_points = []
    source_points = []
    for anchor in anchors:
        bary = np.asarray(anchor["barycentric"], np.float64)
        point_3d = bary @ positions[triangles[int(anchor["triangle_id"])]]
        target_xy, _depth = camera.project(point_3d[None, :])
        target_points.append(target_xy[0])
        source_points.append(anchor["source_xy"])
    candidate_atlas, texture_report, writable = build_face_patch_atlas(
        baseline_rgb, source_rgb, source_alpha, positions, uv, triangles,
        selected_ids, camera, np.asarray(target_points), np.asarray(source_points),
        minimum_alpha=minimum_alpha, tps_regularization=1e-3,
    )
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(candidate_atlas, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("FACE_CANDIDATE_ATLAS_ENCODE_FAILED")
    atlas_path = output_dir / "atlas_face_surface_owned_2048.png"
    glb_path = output_dir / "panda_face_surface_owned_2048.glb"
    atlas_path.write_bytes(encoded.tobytes())
    before_hashes = immutable_buffer_hashes(baseline_glb)
    bind_texture(baseline_glb, glb_path, encoded.tobytes(), textured_triangles=np.ones(len(triangles), bool))
    after_hashes = immutable_buffer_hashes(glb_path)
    if before_hashes != after_hashes:
        raise RuntimeError("FACE_CANDIDATE_GEOMETRY_UV_INDEX_CHANGED")
    non_face_changed = int(np.any(candidate_atlas[~writable] != baseline_rgb[~writable], axis=1).sum())
    if non_face_changed:
        raise RuntimeError("FACE_CANDIDATE_NON_FACE_CHANGED")
    candidate_records = []
    for record in records[:32]:
        row = dict(record)
        row["triangle_ids"] = record["triangle_ids"].astype(int).tolist()
        candidate_records.append(row)
    report = {
        "schema": "face_surface_candidate_v1",
        "classification": "CANDIDATE_REQUIRES_VISUAL_REVIEW",
        "baseline_glb": str(baseline_glb),
        "baseline_glb_sha256": sha256(baseline_glb),
        "baseline_atlas": str(baseline_atlas),
        "baseline_atlas_sha256": sha256(baseline_atlas),
        "source_image": str(source_image),
        "source_image_sha256": sha256(source_image),
        "camera": camera_payload,
        "foreground_pixels": int(foreground.sum()),
        "face_mask_pixels": int(face_mask.sum()),
        "ray_stride": int(ray_stride),
        "ray_count": int(len(layers["pixels_xy"])),
        "hit_count": int(len(layers["triangle_ids"])),
        "maximum_depth_layers": int(np.diff(layers["offsets"]).max()) if len(layers["offsets"]) > 1 else 0,
        "chart_inventory": chart_inventory,
        "selected_patch": {
            key: (value.astype(int).tolist() if isinstance(value, np.ndarray) else value)
            for key, value in selected.items()
        },
        "anchors": anchors,
        "texture": texture_report,
        "non_face_atlas_pixels_changed": non_face_changed,
        "immutable_hashes_before": before_hashes,
        "immutable_hashes_after": after_hashes,
        "output_atlas": str(atlas_path),
        "output_atlas_sha256": sha256(atlas_path),
        "output_glb": str(glb_path),
        "output_glb_sha256": sha256(glb_path),
        "promotion_authorized": False,
    }
    (output_dir / "candidate_patch_scores.json").write_text(json.dumps(candidate_records, indent=2), encoding="utf-8")
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-glb", type=Path, required=True)
    parser.add_argument("--baseline-atlas", type=Path, required=True)
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ray-stride", type=int, default=4)
    parser.add_argument("--minimum-alpha", type=float, default=0.35)
    args = parser.parse_args()
    report = build_candidate(args.baseline_glb, args.baseline_atlas, args.source_image,
                             args.output_dir, ray_stride=args.ray_stride,
                             minimum_alpha=args.minimum_alpha)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
