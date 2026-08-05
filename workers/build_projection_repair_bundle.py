"""Build a CPU face-ID/depth bundle for the bounded panda texture repair.

The bundle uses the current mesh and UVs exactly as supplied.  It is a single
real front source view; no mirrored or synthetic view is eligible to write
semantic pixels.  The face-ID buffer is produced by the same orthographic
triangle rasterisation convention used by the projector.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from mesh_io import face_normals, read_glb
from shaman_texture_views import mask_iou, refine_box, warp_to_frame


def project(vertices: np.ndarray, direction: np.ndarray, ortho: float) -> tuple[np.ndarray, np.ndarray]:
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    screen = np.stack(
        [vertices[:, ua] * flip_u / ortho + 0.5, 0.5 - vertices[:, va] / ortho], axis=1
    )
    depth = -(vertices @ direction)
    return screen, depth


def rasterise_face_ids(
    screen: np.ndarray, depth: np.ndarray, tris: np.ndarray, size: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return nearest face ID and triangle visibility at pixel resolution."""
    px = screen * float(size - 1)
    zbuffer = np.full((size, size), np.inf, np.float64)
    face_id = np.full((size, size), -1, np.int32)
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
        d = w0[inside] * depth[tri[0]] + w1[inside] * depth[tri[1]] + w2[inside] * depth[tri[2]]
        old = zbuffer[ys, xs]
        closer = d < old
        if closer.any():
            yi, xi = ys[closer], xs[closer]
            zbuffer[yi, xi] = d[closer]
            face_id[yi, xi] = int(triangle_id)
    visible = np.zeros(len(tris), dtype=bool)
    visible_ids = np.unique(face_id[face_id >= 0])
    visible[visible_ids] = True
    return face_id, visible


def build_bundle(mesh: Path, front_image: Path, output_npz: Path, report_path: Path, size: int) -> dict:
    positions, _normals, uv, tris = read_glb(mesh)
    if uv is None:
        raise RuntimeError("UV_SOURCE_MISSING")
    image = cv2.imread(str(front_image), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise RuntimeError("FRONT_IMAGE_MISSING")
    if image.shape[0] != size or image.shape[1] != size:
        raise RuntimeError(f"FRONT_IMAGE_SIZE_MISMATCH:{image.shape[:2]} != {(size, size)}")
    positions = positions.astype(np.float64)
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    direction = np.array([0.0, -1.0, 0.0], np.float64)
    ortho = 2.6
    screen, depth = project(verts, direction, ortho)
    face_id, visible = rasterise_face_ids(screen, depth, tris, size)
    silhouette = face_id >= 0
    if image.shape[2] == 4:
        source_bgr = image[:, :, :3]
        source_alpha = image[:, :, 3]
        source_mask = source_alpha > 127
    else:
        source_bgr = image[:, :, :3]
        source_alpha = np.full(image.shape[:2], 255, np.uint8)
        source_mask = source_bgr.min(axis=2) < 245
    source_ys, source_xs = np.nonzero(source_mask)
    if not len(source_xs):
        raise RuntimeError("FRONT_SOURCE_MASK_EMPTY")
    source_box = (
        int(source_xs.min()), int(source_ys.min()),
        int(source_xs.max()) + 1, int(source_ys.max()) + 1,
    )
    mesh_ys, mesh_xs = np.nonzero(silhouette)
    mesh_box = (
        int(mesh_xs.min()), int(mesh_ys.min()),
        int(mesh_xs.max()) + 1, int(mesh_ys.max()) + 1,
    )
    registered_box, affine_iou = refine_box(
        source_mask, silhouette, source_box, mesh_box, size
    )
    registered_bgr = warp_to_frame(
        source_bgr, source_box, registered_box, size, interpolation=cv2.INTER_LINEAR
    )
    registered_alpha = warp_to_frame(
        source_alpha, source_box, registered_box, size, interpolation=cv2.INTER_NEAREST
    )
    registered_path = output_npz.parent / "front.png"
    cv2.imwrite(str(registered_path), np.dstack([registered_bgr, registered_alpha]))
    (output_npz.parent / "view_metadata.json").write_text(
        json.dumps({
            "policy": {"semantic_projection": ["real", "generated"]},
            "views": [{"view": "front", "source_type": "real", "confidence": 1.0}],
        }, indent=2),
        encoding="utf-8",
    )
    normals = face_normals(verts, tris).astype(np.float32)
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        verts=verts.astype(np.float32),
        tris=tris.astype(np.int32),
        uvs=uv[tris].astype(np.float32),
        normals=normals,
        view_names=np.array(["front"]),
        view_locs=np.asarray([direction * (ortho * 3.0)], np.float32),
        ortho_scale=np.float32(ortho),
        vis_front=visible,
        face_id_front=face_id,
    )
    report = {
        "schema": "panda_projection_repair_bundle_v1",
        "mesh": str(mesh),
        "front_image": str(front_image),
        "raster_size": size,
        "camera_direction": direction.tolist(),
        "ortho_scale": ortho,
        "triangles": int(len(tris)),
        "visible_triangles": int(visible.sum()),
        "face_id_pixels": int(np.count_nonzero(face_id >= 0)),
        "face_id_buffer": "nearest_triangle_depth_raster_cpu",
        "source_bbox": list(source_box),
        "mesh_bbox": list(mesh_box),
        "registered_bbox": list(registered_box),
        "registration_affine_iou": round(float(affine_iou), 6),
        "registered_front": str(registered_path),
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    cv2.imwrite(str(output_npz.parent / "face_id_front.png"), np.uint8(np.clip(face_id + 1, 0, 255)))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--front-image", required=True)
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    report = build_bundle(Path(args.mesh), Path(args.front_image), Path(args.output_npz), Path(args.report), args.size)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
