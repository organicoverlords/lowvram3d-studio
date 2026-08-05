"""Texture-only face refinement with independent coverage and density evidence."""
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
from shaman_texture_views import project


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _normalised_points(values: dict, width: int, height: int) -> tuple[list[str], np.ndarray]:
    names = list(values)
    points = np.asarray([[float(values[name][0]) * (width - 1),
                          float(values[name][1]) * (height - 1)] for name in names], dtype=np.float32)
    return names, points


def _barycentric(point: np.ndarray, triangle: np.ndarray) -> np.ndarray:
    a, b, c = triangle
    e1, e2 = b - a, c - a
    den = float(e1[0] * e2[1] - e1[1] * e2[0])
    if abs(den) < 1e-12:
        return np.array([1.0, 0.0, 0.0], dtype=np.float64)
    rel = point - a
    w1 = (rel[0] * e2[1] - rel[1] * e2[0]) / den
    w2 = (e1[0] * rel[1] - e1[1] * rel[0]) / den
    return np.asarray([1.0 - w1 - w2, w1, w2], dtype=np.float64)


def _project_anchor_points(config: dict, verts: np.ndarray, tris: np.ndarray,
                           view_locs: np.ndarray, ortho: float,
                           width: int, height: int) -> tuple[list[str], np.ndarray, dict]:
    anchors = config.get("target_landmark_anchors")
    if not anchors:
        names, points = _normalised_points(config["target_landmarks_normalized"], width, height)
        return names, points, {"mode": "legacy_2d", "anchors": {}}
    direction = np.asarray(view_locs[0], dtype=np.float64)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    screen, _depth = project(verts, direction, ortho)
    names, points, record = list(anchors), [], {}
    for name in names:
        anchor = anchors[name]
        triangle_id = int(anchor["triangle_id"])
        bary = np.asarray(anchor["barycentric"], dtype=np.float64)
        if triangle_id < 0 or triangle_id >= len(tris) or bary.shape != (3,):
            raise RuntimeError(f"FACE_ANCHOR_INVALID:{name}")
        if not np.isfinite(bary).all() or abs(float(bary.sum()) - 1.0) > 1e-4:
            raise RuntimeError(f"FACE_ANCHOR_BARYCENTRIC_INVALID:{name}")
        projected = bary @ screen[tris[triangle_id]]
        points.append(projected * np.asarray([width - 1, height - 1], dtype=np.float64))
        record[name] = {"triangle_id": triangle_id, "barycentric": bary.tolist(),
                        "projected_pixels": [float(projected[0] * (width - 1)),
                                              float(projected[1] * (height - 1))]}
    return names, np.asarray(points, dtype=np.float32), {"mode": "3d_triangle_barycentric", "anchors": record}


def _predict_affine(source_train: np.ndarray, target_train: np.ndarray, point: np.ndarray) -> np.ndarray:
    if len(source_train) >= 3:
        triangulation = Delaunay(source_train)
        simplex = int(triangulation.find_simplex(point))
        if simplex >= 0:
            indices = triangulation.simplices[simplex]
            return _barycentric(point, source_train[indices]) @ target_train[indices]
    # A held-out landmark can lie on the convex hull. Use all remaining anchors for a
    # least-squares affine extrapolation instead of a nearest-three fit that can explode.
    matrix = np.column_stack([source_train, np.ones(len(source_train))])
    coefficients, *_ = np.linalg.lstsq(matrix, target_train, rcond=None)
    return np.asarray([*point, 1.0], dtype=np.float32) @ coefficients


def _leave_one_out(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    errors = []
    for index in range(len(source_points)):
        keep = np.arange(len(source_points)) != index
        errors.append(float(np.linalg.norm(_predict_affine(source_points[keep], target_points[keep], source_points[index]) - target_points[index])))
    return np.asarray(errors, dtype=np.float64)


def _piecewise_affine(source: np.ndarray, source_points: np.ndarray, target_points: np.ndarray,
                      width: int, height: int, supersample: int = 2) -> tuple[np.ndarray, np.ndarray]:
    """Bilinear source sampling at supersample resolution followed by area downsampling."""
    scale = max(int(supersample), 1)
    hi_width, hi_height = width * scale, height * scale
    target_hi = target_points * float(scale)
    result = np.zeros((hi_height, hi_width, 3), np.uint8)
    valid = np.zeros((hi_height, hi_width), np.uint8)
    triangulation = Delaunay(target_hi)
    for simplex in triangulation.simplices:
        src = source_points[simplex].astype(np.float32)
        dst = target_hi[simplex].astype(np.float32)
        lo = np.maximum(np.floor(dst.min(axis=0)).astype(int), 0)
        hi = np.minimum(np.ceil(dst.max(axis=0)).astype(int) + 1, [hi_width, hi_height])
        if np.any(hi <= lo):
            continue
        yy, xx = np.mgrid[lo[1]:hi[1], lo[0]:hi[0]].astype(np.float32)
        flat = np.stack([xx.ravel() + 0.5, yy.ravel() + 0.5], axis=1)
        bary = np.asarray([_barycentric(point, dst) for point in flat])
        inside = np.all(bary >= -1e-4, axis=1)
        if not inside.any():
            continue
        inverse = cv2.invertAffineTransform(cv2.getAffineTransform(src, dst))
        sample = flat @ inverse[:, :2].T + inverse[:, 2]
        good = inside & (sample[:, 0] >= 0) & (sample[:, 0] <= source.shape[1] - 1)
        good &= (sample[:, 1] >= 0) & (sample[:, 1] <= source.shape[0] - 1)
        if not good.any():
            continue
        patch_h, patch_w = hi[1] - lo[1], hi[0] - lo[0]
        map_x = sample[:, 0].reshape(patch_h, patch_w).astype(np.float32)
        map_y = sample[:, 1].reshape(patch_h, patch_w).astype(np.float32)
        patch = cv2.remap(source, map_x, map_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        good_patch = good.reshape(patch_h, patch_w)
        result[lo[1]:hi[1], lo[0]:hi[0]][good_patch] = patch[good_patch]
        valid[lo[1]:hi[1], lo[0]:hi[0]][good_patch] = 255
    if scale == 1:
        return result, valid > 0
    return (cv2.resize(result, (width, height), interpolation=cv2.INTER_AREA),
            cv2.resize(valid, (width, height), interpolation=cv2.INTER_AREA) > 220)


def _atlas_triangle_pixels(uv: np.ndarray, tri: np.ndarray, size: int):
    points = (uv if uv.ndim == 2 else uv[tri]).astype(np.float64)
    points[:, 0] *= size - 1
    points[:, 1] = (1.0 - points[:, 1]) * (size - 1)
    lo = np.maximum(np.floor(points.min(axis=0)).astype(int), 0)
    hi = np.minimum(np.ceil(points.max(axis=0)).astype(int), size - 1)
    if np.any(hi < lo):
        return None
    xs, ys = np.meshgrid(np.arange(lo[0], hi[0] + 1), np.arange(lo[1], hi[1] + 1))
    xs, ys = xs.ravel(), ys.ravel()
    (x0, y0), (x1, y1), (x2, y2) = points
    px, py = xs + 0.5, ys + 0.5
    den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(float(den)) < 1e-12:
        return None
    w0 = ((y1 - y2) * (px - x2) + (x2 - x1) * (py - y2)) / den
    w1 = ((y2 - y0) * (px - x2) + (x0 - x2) * (py - y2)) / den
    w2 = 1.0 - w0 - w1
    inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
    return (xs[inside], ys[inside], w0[inside], w1[inside], w2[inside]) if inside.any() else None


def _component_stats(mask: np.ndarray) -> tuple[dict, np.ndarray]:
    count, labels, stats, _centres = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    components = []
    for index in range(1, count):
        x, y, width, height, area = stats[index]
        components.append({"label": int(index), "x": int(x), "y": int(y), "width": int(width), "height": int(height), "area": int(area)})
    components.sort(key=lambda item: item["area"], reverse=True)
    largest = components[0] if components else {"width": 0, "height": 0, "area": 0}
    data = {"component_count": len(components), "components": components[:32],
            "largest_component_width": int(largest.get("width", 0)),
            "largest_component_height": int(largest.get("height", 0)),
            "largest_component_area": int(largest.get("area", 0)),
            "equivalent_square_texels": round(float(np.sqrt(max(int(mask.sum()), 0))), 3),
            "total_texels": int(mask.sum())}
    return data, labels


def _direct_observation_percent(accepted: np.ndarray, intended: np.ndarray) -> float:
    accepted = np.asarray(accepted, dtype=bool)
    intended = np.asarray(intended, dtype=bool)
    if accepted.shape != intended.shape:
        raise ValueError("accepted and intended masks must have matching shapes")
    return 100.0 * float(np.count_nonzero(accepted & intended)) / max(int(np.count_nonzero(intended)), 1)


def _write_component_map(labels: np.ndarray, path: Path) -> None:
    encoded = np.zeros((*labels.shape, 3), np.uint8)
    for label in np.unique(labels):
        if label <= 0:
            continue
        value = int(label) * 2654435761 & 0xFFFFFF
        encoded[labels == label] = (value & 255, (value >> 8) & 255, (value >> 16) & 255)
    cv2.imwrite(str(path), encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("mesh", "source-image", "source-matte", "npz", "view-report", "basecolor", "output", "report", "diagnostics-dir", "manifest"):
        parser.add_argument(f"--{name}", required=True)
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
    matte_alpha = matte[..., 3] if matte.ndim == 3 and matte.shape[2] == 4 else cv2.cvtColor(matte, cv2.COLOR_BGR2GRAY) if matte.ndim == 3 else matte
    source_valid = matte_alpha > 10
    if source_valid.shape != source.shape[:2]:
        source_valid = cv2.resize(source_valid.astype(np.uint8), (source.shape[1], source.shape[0]), interpolation=cv2.INTER_NEAREST) > 0
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
    sx0, sy0 = max(int(round(source_roi[0] * (source.shape[1] - 1))), 0), max(int(round(source_roi[1] * (source.shape[0] - 1))), 0)
    sx1, sy1 = min(int(round(source_roi[2] * (source.shape[1] - 1))), source.shape[1]), min(int(round(source_roi[3] * (source.shape[0] - 1))), source.shape[0])
    source_crop = source[sy0:sy1, sx0:sx1]
    source_grey = cv2.cvtColor(source_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    source_face_edge_energy_raw = float(np.abs(cv2.Laplacian(source_grey, cv2.CV_32F, ksize=3)).std() / 255.0)
    source_face_edge_energy = float(np.abs(cv2.Laplacian(cv2.GaussianBlur(source_grey, (0, 0), 1.0), cv2.CV_32F, ksize=3)).std() / 255.0)
    source_names, source_points = _normalised_points(config["source_landmarks_normalized"], source.shape[1], source.shape[0])
    target_names, target_points, anchor_record = _project_anchor_points(config, verts, tris, view_locs, ortho, view_width, view_height)
    if source_names != target_names:
        raise RuntimeError("FACE_LANDMARK_NAME_MISMATCH")
    loo_errors = _leave_one_out(source_points, target_points)
    diagnostics = Path(args.diagnostics_dir)
    diagnostics.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(diagnostics / "source_face_crop.png"), cv2.cvtColor(source_crop, cv2.COLOR_RGB2BGR))
    local_view, local_valid = _piecewise_affine(source, source_points, target_points, view_width, view_height, supersample=2)
    target_hull = cv2.convexHull(target_points.astype(np.int32))
    target_region = np.zeros((view_height, view_width), np.uint8)
    cv2.fillConvexPoly(target_region, target_hull, 255)
    roi = np.asarray(config.get("target_roi_normalized", [0.0, 0.0, 1.0, 1.0]))
    rx0, ry0, rx1, ry1 = int(roi[0] * view_width), int(roi[1] * view_height), int(roi[2] * view_width), int(roi[3] * view_height)
    roi_mask = np.zeros_like(target_region)
    roi_mask[max(0, ry0):min(view_height, ry1), max(0, rx0):min(view_width, rx1)] = 255
    target_region = (target_region > 0) & (roi_mask > 0) & (face_id >= 0)
    face_tri_ids = np.unique(face_id[target_region]).astype(np.int32)
    face_tri_ids = face_tri_ids[face_tri_ids >= 0]
    if not face_tri_ids.size:
        raise RuntimeError("FACE_TARGET_TRIANGLES_EMPTY")
    image = cv2.imread(args.basecolor, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"BASECOLOR_UNREADABLE:{args.basecolor}")
    atlas = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.uint8)
    atlas_size = atlas.shape[0]
    intended = np.zeros((atlas_size, atlas_size), bool)
    visible_mask = np.zeros_like(intended)
    accepted = np.zeros_like(intended)
    exact_mask = np.zeros_like(intended)
    tolerance_mask = np.zeros_like(intended)
    reason_masks = {name: np.zeros_like(intended) for name in ("occluded", "face_id_mismatch", "outside_local_warp", "source_alpha_failed", "facing_failed", "outside_frame")}
    density = np.zeros((atlas_size, atlas_size), np.float32)
    density_values, sample_count, matched_count = [], 0, 0
    direction = view_locs[0] / max(float(np.linalg.norm(view_locs[0])), 1e-12)
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    for triangle_id in face_tri_ids:
        entry = _atlas_triangle_pixels(uv[triangle_id], np.arange(3), atlas_size)
        if entry is None:
            continue
        xs, ys, w0, w1, w2 = entry
        intended[ys, xs] = True
        tri_positions = verts[tris[triangle_id]]
        world = w0[:, None] * tri_positions[0] + w1[:, None] * tri_positions[1] + w2[:, None] * tri_positions[2]
        u, v = (world[:, ua] * flip_u) / ortho + 0.5, 0.5 - world[:, va] / ortho
        in_frame = (u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)
        reason_masks["outside_frame"][ys[~in_frame], xs[~in_frame]] = True
        vx = np.clip((u * (view_width - 1)).astype(int), 0, view_width - 1)
        vy = np.clip((v * (view_height - 1)).astype(int), 0, view_height - 1)
        facing_ok = float(normals[triangle_id] @ direction) > 0.15
        if not visible[triangle_id]:
            reason_masks["occluded"][ys[in_frame], xs[in_frame]] = True
        if not facing_ok:
            reason_masks["facing_failed"][ys[in_frame], xs[in_frame]] = True
        visible_mask[ys[in_frame & visible[triangle_id] & facing_ok], xs[in_frame & visible[triangle_id] & facing_ok]] = True
        local_ok = in_frame & local_valid[vy, vx]
        reason_masks["outside_local_warp"][ys[in_frame & ~local_ok], xs[in_frame & ~local_ok]] = True
        source_ok = np.zeros_like(in_frame)
        indices = np.nonzero(in_frame)[0]
        source_ok[indices] = source_valid[np.clip((v[indices] * (source.shape[0] - 1)).astype(int), 0, source.shape[0] - 1), np.clip((u[indices] * (source.shape[1] - 1)).astype(int), 0, source.shape[1] - 1)]
        reason_masks["source_alpha_failed"][ys[in_frame & ~source_ok], xs[in_frame & ~source_ok]] = True
        exact = face_id_matches_within_radius(face_id, vx, vy, int(triangle_id), 0)
        matches = face_id_matches_within_radius(face_id, vx, vy, int(triangle_id), args.face_id_radius)
        reason_masks["face_id_mismatch"][ys[in_frame & visible[triangle_id] & facing_ok & ~matches], xs[in_frame & visible[triangle_id] & facing_ok & ~matches]] = True
        sample_ok = in_frame & visible[triangle_id] & facing_ok & local_ok & source_ok
        sample_count += int(sample_ok.sum())
        matched_count += int((sample_ok & matches).sum())
        accept = sample_ok & matches
        if not accept.any():
            continue
        yi, xi = ys[accept], xs[accept]
        accepted[yi, xi] = True
        atlas[yi, xi] = local_view[vy[accept], vx[accept]]
        exact_mask[ys[accept & exact], xs[accept & exact]] = True
        tolerance_mask[ys[accept & ~exact], xs[accept & ~exact]] = True
        tri_edges = np.cross(tri_positions[1] - tri_positions[0], tri_positions[2] - tri_positions[0])
        density_value = float(np.sqrt(max(int(accept.sum()), 1) / max(float(np.linalg.norm(tri_edges)) * 0.5, 1e-12)))
        density[yi, xi], density_values = density_value, density_values + [density_value]
    if not intended.any() or not accepted.any():
        raise RuntimeError("FACE_ATLAS_FOOTPRINT_EMPTY")
    ring = cv2.dilate(accepted.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & ~accepted
    blurred = cv2.GaussianBlur(atlas, (0, 0), 2.0)
    atlas[ring] = (0.75 * atlas[ring] + 0.25 * blurred[ring]).astype(np.uint8)
    protected_path = diagnostics / "protected_face_mask.png"
    cv2.imwrite(str(protected_path), accepted.astype(np.uint8) * 255)
    cv2.imwrite(str(diagnostics / "intended_face_atlas_mask.png"), intended.astype(np.uint8) * 255)
    cv2.imwrite(str(diagnostics / "visible_face_atlas_mask.png"), visible_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(diagnostics / "accepted_face_atlas_mask.png"), accepted.astype(np.uint8) * 255)
    reason_overlay = np.zeros((*intended.shape, 3), np.uint8)
    colours = {"outside_frame": (180, 80, 220), "outside_local_warp": (180, 80, 220), "occluded": (255, 100, 40), "facing_failed": (255, 170, 40), "source_alpha_failed": (255, 40, 180), "face_id_mismatch": (40, 40, 255)}
    for name, colour in colours.items():
        reason_overlay[reason_masks[name]] = colour
    reason_overlay[accepted] = (40, 210, 40)
    cv2.imwrite(str(diagnostics / "rejection_reason_overlay.png"), reason_overlay)
    id_overlay = np.zeros((*intended.shape, 3), np.uint8)
    id_overlay[exact_mask], id_overlay[tolerance_mask] = (40, 210, 40), (40, 210, 210)
    cv2.imwrite(str(diagnostics / "exact_tolerance_face_id_overlay.png"), id_overlay)
    intended_stats, labels = _component_stats(intended)
    accepted_stats, _accepted_labels = _component_stats(accepted)
    _write_component_map(labels, diagnostics / "face_uv_component_map.png")
    density_image = np.zeros((*density.shape, 3), np.uint8)
    if density.max() > 0:
        density_image[..., 0] = np.clip(density / density.max() * 255, 0, 255).astype(np.uint8)
        density_image[..., 1] = np.clip(255 - density / density.max() * 200, 0, 255).astype(np.uint8)
    cv2.imwrite(str(diagnostics / "texel_density_heatmap.png"), density_image)
    loo_overlay = local_view.copy()
    for name, point, error in zip(target_names, target_points, loo_errors):
        colour = (40, 220, 40) if error <= 8.0 else (40, 40, 255)
        xy = tuple(np.rint(point).astype(int))
        cv2.circle(loo_overlay, xy, 8, colour, 3)
        cv2.putText(loo_overlay, f"{name}:{error:.1f}", (xy[0] + 8, xy[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    cv2.imwrite(str(diagnostics / "landmark_loo_residual_overlay.png"), cv2.cvtColor(loo_overlay, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(diagnostics / "local_warp_preview.png"), cv2.cvtColor(local_view, cv2.COLOR_RGB2BGR))
    provenance_view = np.zeros_like(local_view)
    provenance_view[target_region], provenance_view[target_region & local_valid] = (40, 40, 255), (40, 210, 210)
    provenance_view[target_region & local_valid & (face_id >= 0)] = (40, 210, 40)
    cv2.imwrite(str(diagnostics / "face_provenance_overlay.png"), cv2.cvtColor(provenance_view, cv2.COLOR_RGB2BGR))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(atlas, cv2.COLOR_RGB2BGR))
    density_values = np.asarray(density_values or [0.0], dtype=np.float64)
    intended_area, accepted_area = int(intended.sum()), int(accepted.sum())
    screen, _depth = project(verts, direction, ortho)
    screen_pixels = screen[tris].mean(axis=1) * np.asarray([view_width - 1, view_height - 1])

    def region_area(point: np.ndarray, radius: float) -> int:
        region_ids = face_tri_ids[np.linalg.norm(screen_pixels[face_tri_ids] - point, axis=1) <= radius]
        region = np.zeros_like(accepted)
        for region_id in region_ids:
            region_entry = _atlas_triangle_pixels(uv[region_id], np.arange(3), atlas_size)
            if region_entry is not None:
                region[region_entry[1], region_entry[0]] = True
        return int((region & accepted).sum())

    target_by_name = {name: point for name, point in zip(target_names, target_points)}
    left_eye_area = region_area(target_by_name["left_eye"], 42.0)
    right_eye_area = region_area(target_by_name["right_eye"], 42.0)
    eye_diameter = min(2.0 * np.sqrt(max(left_eye_area, 0) / np.pi),
                       2.0 * np.sqrt(max(right_eye_area, 0) / np.pi))
    beak_mid = (target_by_name["upper_face"] + target_by_name["lower_beak"]) * 0.5
    beak_length = float(np.sqrt(max(region_area(beak_mid, 65.0), 0)))
    report = {
        "schema": "face_texture_refine_v2", "mesh": args.mesh, "source_image": args.source_image, "source_matte": args.source_matte,
        "basecolor_input": args.basecolor, "output": str(output_path), "source_roi_normalized": source_roi.tolist(), "source_roi_pixels": [sx0, sy0, sx1, sy1],
        "source_landmarks_normalized": config["source_landmarks_normalized"], "target_landmark_anchor_projection": anchor_record,
        "target_landmarks_pixels": {name: [float(p[0]), float(p[1])] for name, p in zip(target_names, target_points)},
        "landmark_loo_errors_pixels": {name: float(error) for name, error in zip(target_names, loo_errors)}, "landmark_loo_p50_pixels": float(np.percentile(loo_errors, 50)), "landmark_loo_p95_pixels": float(np.percentile(loo_errors, 95)), "landmark_loo_max_pixels": float(np.max(loo_errors)),
        "selected_face_triangle_ids": face_tri_ids.astype(int).tolist(), "selected_face_triangle_count": int(face_tri_ids.size), "intended_face_texel_count": intended_area, "visible_face_texel_count": int(visible_mask.sum()), "accepted_face_texel_count": accepted_area,
        "direct_face_observation_percent": round(_direct_observation_percent(accepted, intended), 4), "face_id_sample_count": int(sample_count), "face_id_match_count": int(matched_count), "face_id_match_percent": round(100.0 * matched_count / max(sample_count, 1), 4),
        "rejected_intended_texels_by_reason": {name: int(mask.sum()) for name, mask in reason_masks.items()}, "face_components": intended_stats, "accepted_components": accepted_stats,
        "largest_component_width": intended_stats["largest_component_width"], "largest_component_height": intended_stats["largest_component_height"], "largest_component_area": intended_stats["largest_component_area"], "equivalent_square_texels": intended_stats["equivalent_square_texels"], "eye_region_texel_diameter": round(float(eye_diameter), 3), "beak_region_texel_length": round(beak_length, 3),
        "texels_per_world_area": {"p05": float(np.percentile(density_values, 5)), "p50": float(np.percentile(density_values, 50)), "p95": float(np.percentile(density_values, 95))}, "source_face_width_pixels": max(sx1 - sx0, 1), "source_magnification_ratio": round((sx1 - sx0) / max(intended_stats["equivalent_square_texels"], 1), 6),
        "source_face_edge_energy": round(source_face_edge_energy, 6), "source_face_edge_energy_raw": round(source_face_edge_energy_raw, 6), "sampling": {"method": "bilinear", "supersample": 2, "downsample": "area", "nearest_neighbour": False}, "boundary_transition": {"rings": 2, "texels": int(ring.sum()), "core_mask": str(protected_path)},
        "protected_face_mask": str(protected_path), "protected_face_texel_sha256": _sha256_bytes(atlas[accepted].tobytes()), "diagnostics": {name: str(diagnostics / name) for name in ("source_face_crop.png", "intended_face_atlas_mask.png", "visible_face_atlas_mask.png", "accepted_face_atlas_mask.png", "rejection_reason_overlay.png", "exact_tolerance_face_id_overlay.png", "face_uv_component_map.png", "texel_density_heatmap.png", "landmark_loo_residual_overlay.png", "local_warp_preview.png", "face_provenance_overlay.png")},
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    if report["direct_face_observation_percent"] < float(config.get("min_direct_observation_percent", 70.0)):
        raise SystemExit("FACE_DIRECT_OBSERVATION_BELOW_GATE")
    if report["largest_component_width"] < int(config.get("min_largest_component_texels", 192)):
        raise SystemExit("FACE_CONTIGUOUS_RESOLUTION_BELOW_GATE")
    if report["face_id_match_percent"] < float(config.get("min_face_id_match_percent", 99.0)):
        raise SystemExit("FACE_ID_MATCH_BELOW_GATE")
    if report["landmark_loo_p95_pixels"] > float(config.get("max_landmark_loo_p95_pixels", 8.0)):
        raise SystemExit("FACE_LANDMARK_LOO_BELOW_GATE")
    print(f"FACE_REFINE triangles={face_tri_ids.size} direct={report['direct_face_observation_percent']:.2f}% face_id={report['face_id_match_percent']:.2f}% loo_p95={report['landmark_loo_p95_pixels']:.2f}px largest={report['largest_component_width']}x{report['largest_component_height']}", flush=True)


if __name__ == "__main__":
    main()
