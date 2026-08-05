"""Build a source-registered front view and projection NPZ for Pipeline V2.

The old route used one affine bounding-box fit. That aligns the outer silhouette but allows the
face, robe bands, feet and ornaments to drift onto neighbouring geometry. This version keeps the
same proven orthographic/depth-buffer projection, then adds a bounded non-rigid silhouette warp:

1. choose the real front hemisphere by silhouette IoU;
2. optimise the global subject box;
3. calculate a low-resolution signed-distance optical flow from the mesh silhouette back to the
   affine-warped source silhouette;
4. remap the source image with that smooth, capped field;
5. accept the non-rigid candidate only when silhouette IoU improves.

The output contract is unchanged, so workers/raster_project.py remains the projector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mesh_io import read_glb

VISIBLE_PIXEL_FRACTION = 0.35
DEPTH_TOLERANCE_FRACTION = 0.004
FLOW_SIZE = 640
FLOW_MAX_FRACTION = 0.085


def project(vertices: np.ndarray, direction: np.ndarray, ortho: float):
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    u = (vertices[:, ua] * flip_u) / ortho + 0.5
    v = 0.5 - vertices[:, va] / ortho
    depth = -(vertices @ direction)
    return np.stack([u, v], axis=1), depth


def rasterise(screen: np.ndarray, depth: np.ndarray, tris: np.ndarray, size: int):
    px = np.empty_like(screen)
    px[:, 0] = screen[:, 0] * (size - 1)
    px[:, 1] = screen[:, 1] * (size - 1)
    zbuffer = np.full((size, size), np.inf, np.float64)
    triangle_pixels: list[tuple[np.ndarray, np.ndarray, np.ndarray] | None] = []

    for tri in tris:
        a = px[tri]
        x_lo, y_lo = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x_hi, y_hi = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x_hi < x_lo or y_hi < y_lo:
            triangle_pixels.append(None)
            continue
        xs, ys = np.meshgrid(np.arange(x_lo, x_hi + 1), np.arange(y_lo, y_hi + 1))
        xs, ys = xs.ravel(), ys.ravel()
        (x0, y0), (x1, y1), (x2, y2) = a
        den = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(den) < 1e-12:
            triangle_pixels.append(None)
            continue
        fx, fy = xs + 0.5, ys + 0.5
        w0 = ((y1 - y2) * (fx - x2) + (x2 - x1) * (fy - y2)) / den
        w1 = ((y2 - y0) * (fx - x2) + (x0 - x2) * (fy - y2)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            triangle_pixels.append(None)
            continue
        xs, ys = xs[inside], ys[inside]
        d = w0[inside] * depth[tri[0]] + w1[inside] * depth[tri[1]] + w2[inside] * depth[tri[2]]
        np.minimum.at(zbuffer, (ys, xs), d)
        triangle_pixels.append((xs, ys, d))

    span = float(depth.max() - depth.min()) or 1.0
    tolerance = span * DEPTH_TOLERANCE_FRACTION
    visible = np.zeros(len(tris), bool)
    for index, entry in enumerate(triangle_pixels):
        if entry is None:
            continue
        xs, ys, d = entry
        if (d <= zbuffer[ys, xs] + tolerance).mean() >= VISIBLE_PIXEL_FRACTION:
            visible[index] = True
    return visible, np.isfinite(zbuffer)


def subject_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.nonzero(mask)
    if not len(xs):
        raise RuntimeError("empty subject mask")
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def warp_to_frame(
    image, source_box, target_box, size, raster_size=None, interpolation=cv2.INTER_LINEAR
):
    scale = 1.0 if raster_size is None else size / raster_size
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = [v * scale for v in target_box]
    src = np.array([[sx0, sy0], [sx1, sy0], [sx0, sy1]], np.float32)
    dst = np.array([[tx0, ty0], [tx1, ty0], [tx0, ty1]], np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(
        image,
        matrix,
        (size, size),
        flags=interpolation,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.astype(bool), b.astype(bool)
    return float((a & b).sum() / max((a | b).sum(), 1))


def refine_box(source_mask, silhouette, source_box, target_box, size):
    best = list(target_box)
    reference = source_mask.astype(np.uint8) * 255

    def score(box):
        warped = warp_to_frame(reference, source_box, box, size, interpolation=cv2.INTER_NEAREST)
        return mask_iou(warped > 127, silhouette)

    current = score(best)
    span = max(target_box[2] - target_box[0], target_box[3] - target_box[1])
    step = max(int(span * 0.04), 2)
    while step >= 1:
        improved = True
        while improved:
            improved = False
            for axis in range(4):
                for delta in (-step, step):
                    trial = list(best)
                    trial[axis] += delta
                    if trial[2] - trial[0] < 8 or trial[3] - trial[1] < 8:
                        continue
                    value = score(trial)
                    if value > current + 1e-6:
                        best, current, improved = trial, value, True
        step //= 2
    return tuple(best), current


def signed_distance(mask: np.ndarray) -> np.ndarray:
    inside = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
    outside = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
    field = inside - outside
    scale = max(float(np.percentile(np.abs(field), 95)), 1.0)
    return np.clip(field / scale, -1.0, 1.0).astype(np.float32)


def dense_mask_refine(
    source_bgr: np.ndarray,
    source_mask: np.ndarray,
    source_box,
    target_box,
    target_mask_raster: np.ndarray,
    output_size: int,
    raster_size: int,
) -> tuple[np.ndarray, np.ndarray, dict]:
    flow_size = min(FLOW_SIZE, output_size)
    affine_image_small = warp_to_frame(
        source_bgr, source_box, target_box, flow_size, raster_size=raster_size
    )
    affine_mask_small = warp_to_frame(
        source_mask.astype(np.uint8) * 255,
        source_box,
        target_box,
        flow_size,
        raster_size=raster_size,
        interpolation=cv2.INTER_NEAREST,
    ) > 127
    target_small = cv2.resize(
        target_mask_raster.astype(np.uint8),
        (flow_size, flow_size),
        interpolation=cv2.INTER_NEAREST,
    ) > 0
    initial_iou = mask_iou(affine_mask_small, target_small)

    target_field = signed_distance(target_small)
    source_field = signed_distance(affine_mask_small)
    flow = cv2.calcOpticalFlowFarneback(
        target_field,
        source_field,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=55,
        iterations=8,
        poly_n=7,
        poly_sigma=1.5,
        flags=cv2.OPTFLOW_FARNEBACK_GAUSSIAN,
    )
    flow = cv2.GaussianBlur(flow, (0, 0), 5.0)
    magnitude = np.linalg.norm(flow, axis=2)
    cap = flow_size * FLOW_MAX_FRACTION
    scale = np.minimum(1.0, cap / np.maximum(magnitude, 1e-6))
    flow *= scale[..., None]

    yy, xx = np.mgrid[:flow_size, :flow_size].astype(np.float32)
    map_x, map_y = xx + flow[..., 0], yy + flow[..., 1]
    refined_mask_small = cv2.remap(
        affine_mask_small.astype(np.uint8) * 255,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    ) > 127
    refined_iou = mask_iou(refined_mask_small, target_small)
    accepted = refined_iou >= initial_iou + 0.002

    affine_image = warp_to_frame(
        source_bgr, source_box, target_box, output_size, raster_size=raster_size
    )
    affine_mask = warp_to_frame(
        source_mask.astype(np.uint8) * 255,
        source_box,
        target_box,
        output_size,
        raster_size=raster_size,
        interpolation=cv2.INTER_NEAREST,
    )
    summary = {
        "accepted": accepted,
        "affine_iou": round(initial_iou, 6),
        "dense_iou": round(refined_iou, 6),
        "flow_median_pixels": round(float(np.median(magnitude)), 3),
        "flow_p95_pixels": round(float(np.percentile(magnitude, 95)), 3),
        "flow_cap_pixels": round(cap, 3),
    }
    if not accepted:
        return affine_image, affine_mask, summary

    ratio = output_size / flow_size
    flow_full = cv2.resize(flow, (output_size, output_size), interpolation=cv2.INTER_LINEAR) * ratio
    yy, xx = np.mgrid[:output_size, :output_size].astype(np.float32)
    map_x, map_y = xx + flow_full[..., 0], yy + flow_full[..., 1]
    refined_image = cv2.remap(
        affine_image,
        map_x,
        map_y,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    refined_mask = cv2.remap(
        affine_mask,
        map_x,
        map_y,
        cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )
    return refined_image, refined_mask, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raster-size", type=int, default=1536)
    parser.add_argument("--view-size", type=int, default=2048)
    args = parser.parse_args()

    positions, _, uv, tris = read_glb(Path(args.mesh))
    if uv is None:
        raise RuntimeError("projection mesh has no UVs")
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    ortho = float((verts.max(axis=0) - verts.min(axis=0)).max())

    edge1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    edge2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    face_normals = np.cross(edge1, edge2)
    face_normals /= np.maximum(np.linalg.norm(face_normals, axis=1, keepdims=True), 1e-12)

    source = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"could not read {args.source}")
    if source.ndim == 2:
        source = cv2.cvtColor(source, cv2.COLOR_GRAY2BGRA)
    if source.shape[2] == 4:
        source_bgr = source[:, :, :3]
        source_mask = source[:, :, 3] > 127
    else:
        source_bgr = source[:, :, :3]
        source_mask = source_bgr.min(axis=2) < 245

    candidates = {"+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0])}
    scored = {}
    source_box = subject_bbox(source_mask)
    for name, direction in candidates.items():
        screen, depth = project(verts, direction, ortho)
        _, silhouette = rasterise(screen, depth, tris, args.raster_size)
        mesh_box = subject_bbox(silhouette)
        warped = warp_to_frame(
            source_mask.astype(np.uint8) * 255,
            source_box,
            mesh_box,
            args.raster_size,
            interpolation=cv2.INTER_NEAREST,
        )
        scored[name] = {
            "iou": mask_iou(warped > 127, silhouette),
            "silhouette": silhouette,
            "mesh_box": mesh_box,
        }
    front = max(scored, key=lambda key: scored[key]["iou"])
    direction = candidates[front]
    screen, depth = project(verts, direction, ortho)
    visible, silhouette = rasterise(screen, depth, tris, args.raster_size)
    mesh_box, refined_affine_iou = refine_box(
        source_mask, silhouette, source_box, scored[front]["mesh_box"], args.raster_size
    )

    view, view_alpha, dense_report = dense_mask_refine(
        source_bgr,
        source_mask,
        source_box,
        mesh_box,
        silhouette,
        args.view_size,
        args.raster_size,
    )
    rgba = np.dstack([view, view_alpha])
    views_dir = Path(args.views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(views_dir / "front.png"), rgba)
    (views_dir / "view_metadata.json").write_text(
        json.dumps({
            "policy": {"semantic_projection": ["real", "generated"]},
            "views": [{"view": "front", "source_type": "real", "confidence": 1.0}],
        }, indent=2),
        encoding="utf-8",
    )

    cam = direction * (ortho * 3.0)
    np.savez_compressed(
        args.output_npz,
        verts=verts.astype(np.float32),
        tris=tris.astype(np.int32),
        uvs=uv[tris].astype(np.float32),
        normals=face_normals.astype(np.float32),
        view_names=np.array(["front"]),
        view_locs=np.array([cam], np.float32),
        ortho_scale=np.float32(ortho),
        vis_front=visible,
    )

    facing = face_normals @ direction
    final_registration_iou = (
        dense_report["dense_iou"] if dense_report.get("accepted") else dense_report["affine_iou"]
    )
    report = {
        "mesh": args.mesh,
        "source": args.source,
        "front_direction": front,
        "silhouette_iou_bbox_fit": {
            key: round(value["iou"], 6) for key, value in scored.items()
        },
        "silhouette_iou_refined_affine": round(refined_affine_iou, 6),
        "dense_registration": dense_report,
        "final_registration_iou": round(float(final_registration_iou), 6),
        "registration_gate_passed": bool(final_registration_iou >= 0.58),
        "ortho_scale": ortho,
        "centre_offset": [float(v) for v in centre],
        "triangles": int(len(tris)),
        "visible_triangles": int(visible.sum()),
        "visible_percent": round(float(visible.mean() * 100), 3),
        "front_facing_triangles": int((facing > 0.15).sum()),
        "occluded_but_front_facing": int(((facing > 0.15) & ~visible).sum()),
        "mesh_bbox_pixels": list(mesh_box),
        "source_bbox_pixels": list(source_box),
        "raster_size": args.raster_size,
        "view_size": args.view_size,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"VIEWS_BUILT front={front} visible={visible.sum()}/{len(tris)} "
        f"affine_iou={dense_report['affine_iou']} dense_iou={dense_report['dense_iou']} "
        f"dense_accepted={dense_report['accepted']}",
        flush=True,
    )
    raise SystemExit(0 if report["registration_gate_passed"] else 2)


if __name__ == "__main__":
    main()
