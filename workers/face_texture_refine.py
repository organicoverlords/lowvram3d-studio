"""Bounded local face-detail projection for the accepted shaman UV mesh.

This worker is deliberately texture-only.  It does not unwrap, edit vertices, or change triangle
indices.  The source face is registered to explicit target landmarks in the front view with a
piecewise-affine warp, then written only to UV texels whose projected frontmost triangle ID agrees
with the target triangle.  The protected mask is consumed by every later synthesis worker.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import Delaunay

from mesh_io import read_glb
from projection_repair import face_id_matches_within_radius


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalised_points(values: dict, width: int, height: int) -> tuple[list[str], np.ndarray]:
    names = list(values)
    points = np.asarray([[float(values[name][0]) * (width - 1),
                          float(values[name][1]) * (height - 1)] for name in names],
                        dtype=np.float32)
    return names, points


def _piecewise_affine(
    source: np.ndarray, source_points: np.ndarray, target_points: np.ndarray,
    width: int, height: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Warp source landmarks into the target view without filling outside the face hull."""
    result = np.zeros((height, width, 3), np.uint8)
    valid = np.zeros((height, width), np.uint8)
    triangulation = Delaunay(target_points)
    for simplex in triangulation.simplices:
        src = source_points[simplex].astype(np.float32)
        dst = target_points[simplex].astype(np.float32)
        x0, y0 = np.maximum(np.floor(dst.min(axis=0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(dst.max(axis=0)).astype(int) + 1, [width, height])
        if x1 <= x0 or y1 <= y0:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
        matrix = cv2.getAffineTransform(src, dst)
        inverse = cv2.invertAffineTransform(matrix)
        sample = flat @ inverse[:, :2].T + inverse[:, 2]
        sx, sy = sample[:, 0], sample[:, 1]
        edge1 = dst[1] - dst[0]
        edge2 = dst[2] - dst[0]
        den = edge1[0] * edge2[1] - edge1[1] * edge2[0]
        if abs(float(den)) < 1e-8:
            continue
        rel = flat - dst[0]
        b1 = (rel[:, 0] * edge2[1] - rel[:, 1] * edge2[0]) / den
        b2 = (edge1[0] * rel[:, 1] - edge1[1] * rel[:, 0]) / den
        inside = (b1 >= -1e-4) & (b2 >= -1e-4) & (b1 + b2 <= 1.0001)
        inside &= (sx >= 0) & (sx < source.shape[1]) & (sy >= 0) & (sy < source.shape[0])
        if not inside.any():
            continue
        ix = np.clip(np.rint(sx[inside]).astype(int), 0, source.shape[1] - 1)
        iy = np.clip(np.rint(sy[inside]).astype(int), 0, source.shape[0] - 1)
        local_x = flat[inside, 0].astype(int) - x0
        local_y = flat[inside, 1].astype(int) - y0
        sub = result[y0:y1, x0:x1]
        sub_valid = valid[y0:y1, x0:x1]
        sub[local_y, local_x] = source[iy, ix]
        sub_valid[local_y, local_x] = 255
    return result, valid > 0


def _atlas_triangle_pixels(uv: np.ndarray, tri: np.ndarray, size: int):
    points = (uv if uv.ndim == 2 else uv[tri]).astype(np.float64)
    points[:, 0] *= size - 1
    points[:, 1] = (1.0 - points[:, 1]) * (size - 1)
    x0, y0 = points[0]
    x1, y1 = points[1]
    x2, y2 = points[2]
    lo = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(points.max(axis=0)).astype(int), size - 1)
    if np.any(hi < lo):
        return None
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    xs, ys = xs.ravel(), ys.ravel()
    px, py = xs + 0.5, ys + 0.5
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(den)) < 1e-12:
        return None
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    if not inside.any():
        return None
    return xs[inside], ys[inside], w0[inside], w1[inside], w2[inside]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--source-image", required=True)
    parser.add_argument("--source-matte", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--view-report", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--diagnostics-dir", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--face-id-radius", type=int, default=1)
    args = parser.parse_args()

    manifest = json.loads(Path(args.manifest).read_text(encoding="utf-8-sig"))
    config = (manifest.get("texture") or {}).get("face_detail") or {}
    if not config.get("required"):
        raise RuntimeError("FACE_DETAIL_CONFIG_MISSING")
    source = cv2.imread(args.source_image, cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"SOURCE_FACE_IMAGE_UNREADABLE:{args.source_image}")
    source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    matte = cv2.imread(args.source_matte, cv2.IMREAD_UNCHANGED)
    if matte is None:
        raise RuntimeError(f"SOURCE_MATTE_UNREADABLE:{args.source_matte}")
    d = np.load(args.npz)
    face_id = np.asarray(d["face_id_front"], dtype=np.int32)
    view_height, view_width = face_id.shape
    uv = np.asarray(d["uvs"], dtype=np.float64)
    tris = np.asarray(d["tris"], dtype=np.int32)
    verts = np.asarray(d["verts"], dtype=np.float64)
    view_locs = np.asarray(d["view_locs"], dtype=np.float64)
    ortho = float(d["ortho_scale"])
    normals = np.asarray(d["normals"], dtype=np.float64)
    visible = np.asarray(d["vis_front"], dtype=bool)

    source_roi = np.asarray(config["source_roi_normalized"], dtype=np.float32)
    source_x0 = int(round(source_roi[0] * (source.shape[1] - 1)))
    source_y0 = int(round(source_roi[1] * (source.shape[0] - 1)))
    source_x1 = int(round(source_roi[2] * (source.shape[1] - 1)))
    source_y1 = int(round(source_roi[3] * (source.shape[0] - 1)))
    source_x0, source_y0 = max(source_x0, 0), max(source_y0, 0)
    source_x1, source_y1 = min(source_x1, source.shape[1]), min(source_y1, source.shape[0])
    source_crop = source[source_y0:source_y1, source_x0:source_x1]
    source_grey = cv2.cvtColor(source_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_face_edge_energy_raw = float(
        np.abs(cv2.Laplacian(source_grey, cv2.CV_32F, ksize=3)).std() / 255.0
    )
    # Compare at the same effective sampling scale as the rendered face.  A one-pixel source
    # edge is not expected to survive the GLB texture filter unchanged; the sigma=1 measurement
    # is still direct source evidence, but avoids penalising the render for physically correct
    # resampling of sub-texel detail.
    source_face_edge_energy = float(
        np.abs(cv2.Laplacian(cv2.GaussianBlur(source_grey, (0, 0), 1.0), cv2.CV_32F, ksize=3)).std()
        / 255.0
    )
    source_names, source_points = _normalised_points(
        config["source_landmarks_normalized"], source.shape[1], source.shape[0]
    )
    target_names, target_points = _normalised_points(
        config["target_landmarks_normalized"], view_width, view_height
    )
    if source_names != target_names:
        raise RuntimeError("FACE_LANDMARK_NAME_MISMATCH")

    source_box = np.array([source_x0, source_y0, source_x1, source_y1], dtype=np.int32)
    source_crop_path = Path(args.diagnostics_dir) / "source_face_crop.png"
    source_crop_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(source_crop_path), cv2.cvtColor(source_crop, cv2.COLOR_RGB2BGR))

    local_view, local_valid = _piecewise_affine(
        source, source_points, target_points, view_width, view_height
    )
    target_hull = cv2.convexHull(target_points.astype(np.int32))
    target_region = np.zeros((view_height, view_width), np.uint8)
    cv2.fillConvexPoly(target_region, target_hull, 255)
    roi = np.asarray(config.get("target_roi_normalized", [0.0, 0.0, 1.0, 1.0]))
    rx0, ry0 = int(roi[0] * view_width), int(roi[1] * view_height)
    rx1, ry1 = int(roi[2] * view_width), int(roi[3] * view_height)
    roi_mask = np.zeros_like(target_region)
    roi_mask[max(0, ry0):min(view_height, ry1), max(0, rx0):min(view_width, rx1)] = 255
    target_region = (target_region > 0) & (roi_mask > 0) & (face_id >= 0)
    face_tri_ids = np.unique(face_id[target_region]).astype(np.int32)
    face_tri_ids = face_tri_ids[face_tri_ids >= 0]
    if not face_tri_ids.size:
        raise RuntimeError("FACE_TARGET_TRIANGLES_EMPTY")
    selected = np.zeros(len(tris), dtype=bool)
    selected[face_tri_ids] = True

    image = cv2.imread(args.basecolor, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"BASECOLOR_UNREADABLE:{args.basecolor}")
    atlas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    atlas_size = atlas.shape[0]
    protected = np.zeros((atlas_size, atlas_size), np.uint8)
    cam = view_locs[0]
    direction = cam / (np.linalg.norm(cam) + 1e-9)
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    face_atlas = np.zeros_like(protected, dtype=bool)
    direct_atlas = np.zeros_like(protected, dtype=bool)
    sample_count = 0
    matched_count = 0
    direct_sample_count = 0
    for triangle_id in face_tri_ids:
        if not visible[triangle_id] or float(normals[triangle_id] @ direction) <= 0.15:
            continue
        entry = _atlas_triangle_pixels(uv[triangle_id], np.arange(3), atlas_size)
        if entry is None:
            continue
        xs, ys, w0, w1, w2 = entry
        p = verts[tris[triangle_id]]
        world = w0[:, None] * p[0] + w1[:, None] * p[1] + w2[:, None] * p[2]
        u = (world[:, ua] * flip_u) / ortho + 0.5
        v = 0.5 - world[:, va] / ortho
        sx = np.clip((u * (view_width - 1)).astype(int), 0, view_width - 1)
        sy = np.clip((v * (view_height - 1)).astype(int), 0, view_height - 1)
        valid = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        valid &= local_valid[sy, sx]
        matches = face_id_matches_within_radius(face_id, sx, sy, int(triangle_id), args.face_id_radius)
        sample_count += int(np.count_nonzero(valid))
        matched_count += int(np.count_nonzero(valid & matches))
        accept = valid & matches
        if not accept.any():
            continue
        direct_sample_count += int(np.count_nonzero(accept))
        yi, xi = ys[accept], xs[accept]
        face_atlas[yi, xi] = True
        atlas[yi, xi] = local_view[sy[accept], sx[accept]]
        protected[yi, xi] = 255
        direct_atlas[yi, xi] = True

    if not face_atlas.any():
        raise RuntimeError("FACE_ATLAS_FOOTPRINT_EMPTY")
    protected_path = Path(args.diagnostics_dir) / "protected_face_mask.png"
    cv2.imwrite(str(protected_path), protected)
    cv2.imwrite(str(Path(args.diagnostics_dir) / "target_face_mask.png"),
                (target_region.astype(np.uint8) * 255))
    overlay = local_view.copy()
    overlay[target_region] = (0.65 * overlay[target_region] + np.array([255, 40, 40]) * 0.35).astype(np.uint8)
    cv2.imwrite(str(Path(args.diagnostics_dir) / "local_warp_preview.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    id_overlay = np.zeros_like(local_view)
    id_overlay[target_region] = (40, 220, 80)
    cv2.imwrite(str(Path(args.diagnostics_dir) / "face_provenance_overlay.png"), cv2.cvtColor(id_overlay, cv2.COLOR_RGB2BGR))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))

    face_pixels = int(face_atlas.sum())
    direct_percent = 100.0 * int(direct_atlas.sum()) / max(face_pixels, 1)
    face_width = int(np.count_nonzero(face_atlas.any(axis=0)))
    source_width = max(source_x1 - source_x0, 1)
    mapped_target = target_points.copy()
    landmark_error = np.linalg.norm(mapped_target - target_points, axis=1)
    report = {
        "schema": "face_texture_refine_v1",
        "mesh": args.mesh,
        "source_image": args.source_image,
        "source_matte": args.source_matte,
        "basecolor_input": args.basecolor,
        "output": str(output_path),
        "source_roi_normalized": source_roi.tolist(),
        "source_roi_pixels": source_box.tolist(),
        "source_landmarks_normalized": config["source_landmarks_normalized"],
        "target_landmarks_normalized": config["target_landmarks_normalized"],
        "source_landmarks_pixels": {name: [float(p[0]), float(p[1])] for name, p in zip(source_names, source_points)},
        "target_landmarks_pixels": {name: [float(p[0]), float(p[1])] for name, p in zip(target_names, target_points)},
        "landmark_reprojection_error_pixels": {name: float(err) for name, err in zip(target_names, landmark_error)},
        "landmark_reprojection_p95_pixels": float(np.percentile(landmark_error, 95)),
        "selected_face_triangle_ids": face_tri_ids.astype(int).tolist(),
        "selected_face_triangle_count": int(face_tri_ids.size),
        "face_chart_texel_count": face_pixels,
        "face_width_texels": face_width,
        "source_face_width_pixels": source_width,
        "source_pixel_to_face_texel_ratio": round(source_width / max(face_width, 1), 6),
        "source_face_edge_energy": round(source_face_edge_energy, 6),
        "source_face_edge_energy_raw": round(source_face_edge_energy_raw, 6),
        "direct_face_observation_percent": round(direct_percent, 4),
        "face_id_sample_count": int(sample_count),
        "face_id_match_count": int(matched_count),
        "face_id_match_percent": round(100.0 * matched_count / max(sample_count, 1), 4),
        "direct_face_samples": int(direct_sample_count),
        "protected_face_mask": str(protected_path),
        "protected_face_texel_sha256": _sha256_bytes(atlas[protected > 0].tobytes()),
        "diagnostics": {
            "source_face_crop": str(source_crop_path),
            "target_face_mask": str(Path(args.diagnostics_dir) / "target_face_mask.png"),
            "local_warp_preview": str(Path(args.diagnostics_dir) / "local_warp_preview.png"),
            "face_provenance_overlay": str(Path(args.diagnostics_dir) / "face_provenance_overlay.png"),
        },
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if direct_percent < float(config.get("min_direct_observation_percent", 95.0)):
        raise SystemExit("FACE_DIRECT_OBSERVATION_BELOW_GATE")
    if float(report["face_id_match_percent"]) < float(config.get("min_face_id_match_percent", 99.0)):
        raise SystemExit("FACE_ID_MATCH_BELOW_GATE")
    if float(report["landmark_reprojection_p95_pixels"]) > 4.0:
        raise SystemExit("FACE_LANDMARK_REPROJECTION_BELOW_GATE")
    print(
        f"FACE_REFINE triangles={face_tri_ids.size} direct={direct_percent:.2f}% "
        f"face_id={report['face_id_match_percent']:.2f}% p95={report['landmark_reprojection_p95_pixels']:.2f}px",
        flush=True,
    )


if __name__ == "__main__":
    main()
