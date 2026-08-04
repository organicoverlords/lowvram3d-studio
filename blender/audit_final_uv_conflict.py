"""Audit positive-area UV consumers in the imported final GLB.

This is a bounded diagnostic: it keeps every selected front/rear consumer pair,
instead of using a last-writer ownership map. It is intentionally read-only.
"""
import argparse
import json
import math
import time
from pathlib import Path

import bpy
import numpy as np


def reset():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def clip(subject, edge_a, edge_b):
    out = []
    if not subject:
        return out

    def inside(p):
        return (edge_b[0] - edge_a[0]) * (p[1] - edge_a[1]) - (edge_b[1] - edge_a[1]) * (p[0] - edge_a[0]) >= -1e-12

    def intersect(a, b):
        dx, dy = b[0] - a[0], b[1] - a[1]
        ex, ey = edge_b[0] - edge_a[0], edge_b[1] - edge_a[1]
        den = dx * ey - dy * ex
        if abs(den) < 1e-20:
            return [float(b[0]), float(b[1])]
        t = ((edge_a[0] - a[0]) * ey - (edge_a[1] - a[1]) * ex) / den
        return [float(a[0] + t * dx), float(a[1] + t * dy)]

    prev = subject[-1]
    for cur in subject:
        if inside(cur):
            if not inside(prev):
                out.append(intersect(prev, cur))
            out.append(cur)
        elif inside(prev):
            out.append(intersect(prev, cur))
        prev = cur
    return out


def area(poly):
    if len(poly) < 3:
        return 0.0
    return abs(sum(poly[i][0] * poly[(i + 1) % len(poly)][1] - poly[(i + 1) % len(poly)][0] * poly[i][1] for i in range(len(poly))) * 0.5)


def overlap(a, b):
    if a[:, 0].max() < b[:, 0].min() or b[:, 0].max() < a[:, 0].min() or a[:, 1].max() < b[:, 1].min() or b[:, 1].max() < a[:, 1].min():
        return 0.0
    poly = [[float(v[0]), float(v[1])] for v in a]
    for i in range(3):
        poly = clip(poly, b[i], b[(i + 1) % 3])
        if not poly:
            return 0.0
    return area(poly)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--front-ids", required=True)
    parser.add_argument("--rear-ids", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--grid-size", type=int, default=512)
    parser.add_argument("--max-candidates", type=int, default=2000000)
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()

    reset()
    bpy.ops.import_scene.gltf(filepath=args.input)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("no mesh imported")
    obj = max(meshes, key=lambda o: len(o.data.polygons))
    mesh = obj.data
    uv_layer = mesh.uv_layers.active
    if uv_layer is None:
        raise RuntimeError("mesh has no active UV layer")
    triangles = np.asarray([[uv_layer.data[i].uv[:] for i in p.loop_indices] for p in mesh.polygons], dtype=np.float64)
    n = len(triangles)
    front_ids = np.unique(np.load(args.front_ids).astype(np.int64))
    rear_ids = np.unique(np.load(args.rear_ids).astype(np.int64))
    front_ids = front_ids[(front_ids >= 0) & (front_ids < n)]
    rear_ids = rear_ids[(rear_ids >= 0) & (rear_ids < n)]
    signed = 0.5 * ((triangles[:, 1, 0] - triangles[:, 0, 0]) * (triangles[:, 2, 1] - triangles[:, 0, 1]) - (triangles[:, 2, 0] - triangles[:, 0, 0]) * (triangles[:, 1, 1] - triangles[:, 0, 1]))
    valid = np.isfinite(triangles).all(axis=(1, 2)) & (np.abs(signed) > 1e-12)
    front_ids = front_ids[valid[front_ids]]
    rear_ids = rear_ids[valid[rear_ids]]
    grid = max(16, args.grid_size)
    low = np.clip(np.floor(np.nanmin(triangles, axis=1) * grid).astype(np.int64), 0, grid - 1)
    high = np.clip(np.floor(np.nanmax(triangles, axis=1) * grid).astype(np.int64), 0, grid - 1)
    buckets = {}
    for tid in front_ids.tolist():
        for x in range(int(low[tid, 0]), int(high[tid, 0]) + 1):
            for y in range(int(low[tid, 1]), int(high[tid, 1]) + 1):
                buckets.setdefault(x * grid + y, []).append(int(tid))
    pairs = []
    seen = set()
    for tid in rear_ids.tolist():
        candidates = set()
        for x in range(int(low[tid, 0]), int(high[tid, 0]) + 1):
            for y in range(int(low[tid, 1]), int(high[tid, 1]) + 1):
                candidates.update(buckets.get(x * grid + y, ()))
        for fid in candidates:
            pair = (int(fid), int(tid))
            if pair not in seen:
                seen.add(pair)
                pairs.append(pair)
                if len(pairs) > args.max_candidates:
                    raise RuntimeError("candidate cap exceeded")

    started = time.monotonic()
    positives = []
    for fid, rid in pairs:
        if time.monotonic() - started > args.timeout:
            raise RuntimeError("exact test timed out")
        ov = overlap(triangles[fid], triangles[rid])
        if ov > 1e-12:
            positives.append({"front_triangle": fid, "rear_triangle": rid, "overlap_area_uv": ov, "overlap_texels_equivalent_4096": ov * 4096.0 * 4096.0})
    report = {
        "schema": "final_glb_uv_consumer_conflict_v1",
        "input": str(Path(args.input)),
        "triangle_count": n,
        "front_consumer_count": int(len(front_ids)),
        "rear_consumer_count": int(len(rear_ids)),
        "candidate_pair_count": len(pairs),
        "positive_overlap_pair_count": len(positives),
        "positive_overlap_total_texels_equivalent_4096": sum(x["overlap_texels_equivalent_4096"] for x in positives),
        "conflicting_front_triangle_ids": sorted({x["front_triangle"] for x in positives}),
        "conflicting_rear_triangle_ids": sorted({x["rear_triangle"] for x in positives}),
        "positive_pairs": positives[:20000],
        "positive_pairs_truncated": len(positives) > 20000,
        "mesh_object": obj.name,
        "uv_layer": uv_layer.name,
        "elapsed_seconds": time.monotonic() - started,
    }
    out = Path(args.report)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("triangle_count", "candidate_pair_count", "positive_overlap_pair_count", "positive_overlap_total_texels_equivalent_4096", "elapsed_seconds")}, indent=2))


if __name__ == "__main__":
    main()
