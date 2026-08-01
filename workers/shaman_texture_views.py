"""Stage 6, step 1: build the projection NPZ and the canonical view for the shaman.

Feeds workers/raster_project.py without changing a line of it. That projector assumes each view is
an orthographic render of a mesh centred on the origin, framed by a single `ortho_scale`. The
source illustration is not that - it is a portrait crop with its own framing - so rather than bend
the projector's maths to the image, the image is warped into the projector's frame here. The
alignment is measured from the silhouettes, not assumed from the generator's conventions.

Visibility is a real depth buffer. It is what stops the staff painting the robe and the beak
painting the antlers: a triangle contributes colour only where it is genuinely the frontmost
surface, so occluded geometry never receives the pixels of whatever is standing in front of it.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np

COMPONENT_SIZE = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
COMPONENT_DTYPE = {5121: "<u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
# A triangle counts as visible when this share of its covered pixels are the frontmost surface.
VISIBLE_PIXEL_FRACTION = 0.35
DEPTH_TOLERANCE_FRACTION = 0.004


def read_glb(path: Path):
    data = path.read_bytes()
    offset, chunk_json, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            chunk_json = json.loads(payload)
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length

    def accessor(index):
        acc = chunk_json["accessors"][index]
        view = chunk_json["bufferViews"][acc["bufferView"]]
        count, width = acc["count"], TYPE_COUNT[acc["type"]]
        item = COMPONENT_SIZE[acc["componentType"]] * width
        stride = view.get("byteStride") or item
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = np.frombuffer(binary, np.uint8, count=stride * (count - 1) + item, offset=start)
        if stride != item:
            raw = np.lib.stride_tricks.as_strided(raw, (count, item), (stride, 1)).copy()
        return raw.reshape(-1).view(COMPONENT_DTYPE[acc["componentType"]]).reshape(count, width)

    primitive = chunk_json["meshes"][0]["primitives"][0]
    attributes = primitive["attributes"]
    positions = accessor(attributes["POSITION"]).astype(np.float64)
    uv = accessor(attributes["TEXCOORD_0"]).astype(np.float64)
    indices = accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64)
    return positions, uv, indices


def project(vertices: np.ndarray, direction: np.ndarray, ortho: float):
    """Reproduce raster_project.py's orthographic mapping exactly, in normalised [0,1] screen space."""
    axis = int(np.argmax(np.abs(direction)))
    ua, va = (0, 2) if axis == 1 else ((1, 2) if axis == 0 else (0, 1))
    flip_u = -1.0 if direction[axis] > 0 else 1.0
    u = (vertices[:, ua] * flip_u) / ortho + 0.5
    v = 0.5 - vertices[:, va] / ortho
    depth = -(vertices @ direction)  # smaller is nearer the camera
    return np.stack([u, v], axis=1), depth


def rasterise(screen: np.ndarray, depth: np.ndarray, tris: np.ndarray, size: int):
    """Z-buffer the mesh, then decide per-triangle visibility from that buffer."""
    px = np.empty_like(screen)
    px[:, 0] = screen[:, 0] * (size - 1)
    px[:, 1] = screen[:, 1] * (size - 1)

    zbuffer = np.full((size, size), np.inf, np.float64)
    triangle_pixels = []
    for index, tri in enumerate(tris):
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
        frontmost = d <= zbuffer[ys, xs] + tolerance
        if frontmost.mean() >= VISIBLE_PIXEL_FRACTION:
            visible[index] = True
    silhouette = np.isfinite(zbuffer)
    return visible, silhouette


def subject_bbox(mask: np.ndarray):
    ys, xs = np.nonzero(mask)
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--source", required=True, help="matted RGBA source illustration")
    parser.add_argument("--output-npz", required=True)
    parser.add_argument("--views-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--raster-size", type=int, default=1536)
    parser.add_argument("--view-size", type=int, default=2048)
    args = parser.parse_args()

    positions, uv, tris = read_glb(Path(args.mesh))
    centre = (positions.min(axis=0) + positions.max(axis=0)) * 0.5
    verts = positions - centre
    ortho = float((verts.max(axis=0) - verts.min(axis=0)).max())

    edge1 = verts[tris[:, 1]] - verts[tris[:, 0]]
    edge2 = verts[tris[:, 2]] - verts[tris[:, 0]]
    face_normals = np.cross(edge1, edge2)
    lengths = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals = face_normals / np.maximum(lengths, 1e-12)

    source = cv2.imread(args.source, cv2.IMREAD_UNCHANGED)
    if source is None:
        raise RuntimeError(f"could not read {args.source}")
    if source.shape[2] == 4:
        source_alpha = source[:, :, 3].astype(np.float32) / 255.0
        source_rgb = source[:, :, :3]
    else:
        # The matte writes background to pure white rather than carrying alpha.
        source_rgb = source
        source_alpha = (source.min(axis=2) < 245).astype(np.float32)
    source_mask = source_alpha > 0.5

    # Choose the facing direction by silhouette agreement rather than by trusting a generator
    # convention: the staff and its ring are strongly off-centre, so the wrong hemisphere scores
    # visibly worse even though the body alone is nearly symmetric.
    candidates = {"+z": np.array([0.0, 0.0, 1.0]), "-z": np.array([0.0, 0.0, -1.0])}
    scored = {}
    for name, direction in candidates.items():
        screen, depth = project(verts, direction, ortho)
        _, silhouette = rasterise(screen, depth, tris, args.raster_size)
        mesh_box = subject_bbox(silhouette)
        source_box = subject_bbox(source_mask)
        warped = warp_to_frame(source_mask.astype(np.uint8) * 255, source_box, mesh_box, args.raster_size)
        overlap = (warped > 127) & silhouette
        union = (warped > 127) | silhouette
        scored[name] = {
            "iou": float(overlap.sum() / max(union.sum(), 1)),
            "silhouette": silhouette,
            "mesh_box": mesh_box,
            "source_box": source_box,
        }
    front = max(scored, key=lambda k: scored[k]["iou"])
    direction = candidates[front]
    print(f"FRONT_DIRECTION {front} iou=" + " ".join(f"{k}={v['iou']:.4f}" for k, v in scored.items()), flush=True)

    screen, depth = project(verts, direction, ortho)
    visible, silhouette = rasterise(screen, depth, tris, args.raster_size)

    source_box = scored[front]["source_box"]
    mesh_box, refined_iou = refine_box(
        source_mask, silhouette, source_box, scored[front]["mesh_box"], args.raster_size
    )
    print(f"ALIGNMENT iou {scored[front]['iou']:.4f} -> {refined_iou:.4f} box={mesh_box}", flush=True)
    view = warp_to_frame(source_rgb, source_box, mesh_box, args.view_size, raster_size=args.raster_size)
    view_alpha = warp_to_frame(
        (source_mask.astype(np.uint8) * 255), source_box, mesh_box, args.view_size, raster_size=args.raster_size
    )
    rgba = np.dstack([view, view_alpha])

    views_dir = Path(args.views_dir)
    views_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(views_dir / "front.png"), rgba)

    metadata = {
        "policy": {"semantic_projection": ["real", "generated"]},
        "views": [{"view": "front", "source_type": "real", "confidence": 1.0}],
    }
    (views_dir / "view_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

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
    report = {
        "mesh": args.mesh,
        "source": args.source,
        "front_direction": front,
        "silhouette_iou_bbox_fit": {k: round(v["iou"], 4) for k, v in scored.items()},
        "silhouette_iou_refined": round(refined_iou, 4),
        "ortho_scale": ortho,
        "centre_offset": [float(v) for v in centre],
        "triangles": int(len(tris)),
        "visible_triangles": int(visible.sum()),
        "visible_percent": round(float(visible.mean() * 100), 2),
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
        f"VIEWS_BUILT visible={visible.sum()}/{len(tris)} "
        f"({visible.mean()*100:.1f}%) occluded_front_facing={report['occluded_but_front_facing']}",
        flush=True,
    )


def refine_box(source_mask, silhouette, source_box, target_box, size):
    """Nudge the target box to maximise silhouette IoU.

    Matching bounding boxes alone lets one outlying element - the staff ring, an ornament swinging
    wide - set the whole scale, which then slides every other feature off its geometry. Optimising
    the overlap itself keeps the body, head and staff registered instead of just the extremes.
    """
    best = list(target_box)
    reference = source_mask.astype(np.uint8) * 255

    def score(box):
        warped = warp_to_frame(reference, source_box, box, size)
        hit = warped > 127
        return float((hit & silhouette).sum() / max((hit | silhouette).sum(), 1))

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
                        best, current = trial, value
                        improved = True
        step //= 2
    return tuple(best), current


def warp_to_frame(image, source_box, target_box, size, raster_size=None):
    """Map the source subject bounding box onto the mesh silhouette bounding box.

    Independent x and y scaling on purpose: generation drifts the aspect ratio slightly, and forcing
    a single uniform scale leaves the hanging ornaments hanging off the wrong part of the mesh.
    """
    scale = 1.0 if raster_size is None else size / raster_size
    sx0, sy0, sx1, sy1 = source_box
    tx0, ty0, tx1, ty1 = [v * scale for v in target_box]
    src = np.array([[sx0, sy0], [sx1, sy0], [sx0, sy1]], np.float32)
    dst = np.array([[tx0, ty0], [tx1, ty0], [tx0, ty1]], np.float32)
    matrix = cv2.getAffineTransform(src, dst)
    return cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR, borderValue=0)


if __name__ == "__main__":
    main()
