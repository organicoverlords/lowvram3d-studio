"""Build deterministic CPU triangle evidence for arbitrary semantic review views."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

from mesh_io import read_glb


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def unit(x):
    x = np.asarray(x, np.float64)
    return x / max(float(np.linalg.norm(x)), 1e-12)


def rasterise(screen, depth, verts, normals, tris, size):
    px = screen * float(size - 1)
    z = np.full((size, size), np.inf, np.float64)
    face = np.full((size, size), -1, np.int32)
    bary = np.zeros((size, size, 3), np.float32)
    pos = np.zeros((size, size, 3), np.float32)
    norm = np.zeros((size, size, 3), np.float32)
    for tid, tri in enumerate(tris):
        a = px[tri]
        x0, y0 = np.maximum(np.floor(a.min(0)).astype(int), 0)
        x1, y1 = np.minimum(np.ceil(a.max(0)).astype(int), size - 1)
        if x1 < x0 or y1 < y0:
            continue
        xs, ys = np.meshgrid(np.arange(x0, x1 + 1), np.arange(y0, y1 + 1))
        (ax, ay), (bx, by), (cx, cy) = a
        den = (by - cy) * (ax - cx) + (cx - bx) * (ay - cy)
        if abs(den) < 1e-12:
            continue
        fx, fy = xs + 0.5, ys + 0.5
        w0 = ((by - cy) * (fx - cx) + (cx - bx) * (fy - cy)) / den
        w1 = ((cy - ay) * (fx - cx) + (ax - cx) * (fy - cy)) / den
        w2 = 1.0 - w0 - w1
        inside = (w0 >= -1e-4) & (w1 >= -1e-4) & (w2 >= -1e-4)
        if not inside.any():
            continue
        xs, ys = xs[inside], ys[inside]
        weights = np.stack([w0[inside], w1[inside], w2[inside]], 1)
        depth_value = weights @ depth[tri]
        closer = depth_value < z[ys, xs]
        if not closer.any():
            continue
        xs, ys, weights, depth_value = xs[closer], ys[closer], weights[closer], depth_value[closer]
        z[ys, xs] = depth_value
        face[ys, xs] = tid
        bary[ys, xs] = weights
        pos[ys, xs] = weights @ verts[tri]
        norm[ys, xs] = weights @ normals[tri]
    visible = face >= 0
    norm[visible] /= np.maximum(np.linalg.norm(norm[visible], axis=1, keepdims=True), 1e-12)
    return face, bary, z, pos, norm


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--views", required=True, help="JSON list or object containing view camera contracts")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=512)
    args = parser.parse_args()
    mesh_path = Path(args.mesh)
    positions, normals, uv, tris = read_glb(mesh_path)
    positions = positions.astype(np.float64)
    normals = normals.astype(np.float64)
    uv = uv.astype(np.float64)
    tris = tris.astype(np.int64)
    raw = json.loads(Path(args.views).read_text(encoding="utf-8"))
    views = raw.get("views", raw) if isinstance(raw, dict) else raw
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item in views:
        name = str(item["view_name"] if "view_name" in item else item["name"])
        camera = item["camera"]
        direction = unit(camera["direction"])
        up = unit(camera.get("up", [0.0, 0.0, 1.0]))
        right = unit(np.cross(direction, up))
        up = unit(np.cross(right, direction))
        span = float(camera.get("ortho_scale", camera.get("projection_span", 1.0)))
        screen = np.stack([positions @ right / span + 0.5, 0.5 - positions @ up / span], axis=1)
        depth = positions @ direction
        face, bary, z, pos, norm = rasterise(screen, depth, positions, normals, tris, args.resolution)
        visible = face >= 0
        pixel_uv = np.full((args.resolution, args.resolution, 2), -1.0, np.float32)
        if visible.any():
            pixel_uv[visible] = np.einsum("nc,ncd->nd", bary[visible], uv[tris[face[visible]]]).astype(np.float32)
        alpha = np.zeros((args.resolution, args.resolution), np.float32)
        image_path = item.get("image")
        if image_path:
            image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
            if image is not None:
                if image.shape[:2] != alpha.shape:
                    image = cv2.resize(image, (args.resolution, args.resolution), interpolation=cv2.INTER_AREA)
                alpha = image[:, :, 3].astype(np.float32) / 255.0 if image.ndim == 3 and image.shape[2] >= 4 else np.ones_like(alpha)
        semantic = str(item.get("source_class", "UNKNOWN"))
        source_class = np.full((args.resolution, args.resolution), semantic, dtype="U24")
        npz_path = out / f"view_evidence_{name}.npz"
        np.savez_compressed(npz_path, triangle_id=face, barycentric=bary, depth=z, visible=visible, normal_facing=np.clip(norm @ (-direction), -1.0, 1.0), uv=pixel_uv, world_position=pos, source_alpha=alpha, semantic_source_class=source_class, camera_hash=hashlib.sha256(json.dumps(camera, sort_keys=True).encode()).hexdigest(), mesh_hash=sha256(mesh_path))
        manifest.append({"view_name": name, "path": str(npz_path), "camera_hash": hashlib.sha256(json.dumps(camera, sort_keys=True).encode()).hexdigest(), "mesh_hash": sha256(mesh_path), "visible_pixels": int(visible.sum()), "semantic_source_class": semantic})
    manifest_path = out / "view_evidence_manifest.json"
    manifest_path.write_text(json.dumps({"schema": "view_evidence_v1", "mesh": str(mesh_path), "resolution": args.resolution, "views": manifest, "backend": "numpy_cpu_exact_texel_center"}, indent=2), encoding="utf-8")
    Path(args.report).write_text(manifest_path.read_text(encoding="utf-8"), encoding="utf-8")
    print(f"VIEW_EVIDENCE views={len(manifest)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
