"""Stage 6 diagnostic: find base-colour discontinuities across UV chart boundaries.

A seam is visible when two triangles that touch in 3D carry very different colours in the atlas
because they sit in different charts. This measures exactly that: for every pair of triangles
sharing a position-welded edge but landing in different UV charts, compare the colour each side
samples at the shared edge. Large deltas are the seams a viewer would notice.

Reporting the count alone would be misleading - a few sharp deltas on genuinely different materials
(bone meeting cloth) are correct - so the output also renders where the worst offenders sit.
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
WELD = 4e-4


def read_glb(path: Path):
    data = path.read_bytes()
    offset, meta, binary = 12, None, None
    while offset < len(data):
        length, kind = struct.unpack_from("<II", data, offset)
        payload = data[offset + 8 : offset + 8 + length]
        if kind == 0x4E4F534A:
            meta = json.loads(payload)
        elif kind == 0x004E4942:
            binary = payload
        offset += 8 + length

    def accessor(index):
        acc = meta["accessors"][index]
        view = meta["bufferViews"][acc["bufferView"]]
        count, width = acc["count"], TYPE_COUNT[acc["type"]]
        item = COMPONENT_SIZE[acc["componentType"]] * width
        stride = view.get("byteStride") or item
        start = view.get("byteOffset", 0) + acc.get("byteOffset", 0)
        raw = np.frombuffer(binary, np.uint8, count=stride * (count - 1) + item, offset=start)
        if stride != item:
            raw = np.lib.stride_tricks.as_strided(raw, (count, item), (stride, 1)).copy()
        return raw.reshape(-1).view(COMPONENT_DTYPE[acc["componentType"]]).reshape(count, width)

    primitive = meta["meshes"][0]["primitives"][0]
    return (accessor(primitive["attributes"]["POSITION"]).astype(np.float64),
            accessor(primitive["attributes"]["TEXCOORD_0"]).astype(np.float64),
            accessor(primitive["indices"]).reshape(-1, 3).astype(np.int64))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--delta-threshold", type=float, default=0.20)
    args = parser.parse_args()

    positions, uv, tris = read_glb(Path(args.mesh))
    atlas = cv2.cvtColor(cv2.imread(args.basecolor, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    size = atlas.shape[0]

    welded = np.unique(np.round(positions / WELD).astype(np.int64), axis=0, return_inverse=True)[1]

    # Index every undirected welded edge to the (triangle, corner-pair) that produced it.
    edges: dict[tuple[int, int], list[tuple[int, int, int]]] = {}
    for index, tri in enumerate(tris):
        for a, b in ((0, 1), (1, 2), (2, 0)):
            key = (int(min(welded[tri[a]], welded[tri[b]])), int(max(welded[tri[a]], welded[tri[b]])))
            edges.setdefault(key, []).append((index, a, b))

    def sample(tri_index: int, corner_a: int, corner_b: int) -> np.ndarray:
        """Colour just inside the triangle at the midpoint of one of its edges."""
        tri = tris[tri_index]
        edge_mid = (uv[tri[corner_a]] + uv[tri[corner_b]]) * 0.5
        centroid = uv[tri].mean(axis=0)
        point = edge_mid + (centroid - edge_mid) * 0.30
        x = int(np.clip(point[0] * (size - 1), 0, size - 1))
        # glTF convention: v=0 is the first row. The atlas has already been converted out of
        # raster_project's inverted layout by atlas_to_gltf_convention.py.
        y = int(np.clip(point[1] * (size - 1), 0, size - 1))
        return atlas[y, x]

    deltas = []
    seam_points = []
    for key, users in edges.items():
        if len(users) != 2:
            continue
        (ta, aa, ab), (tb, ba, bb) = users
        uv_a = (uv[tris[ta][aa]] + uv[tris[ta][ab]]) * 0.5
        uv_b = (uv[tris[tb][ba]] + uv[tris[tb][bb]]) * 0.5
        if np.linalg.norm(uv_a - uv_b) < 1.5 / size:
            continue  # same chart: not a seam
        delta = float(np.abs(sample(ta, aa, ab) - sample(tb, ba, bb)).max())
        deltas.append(delta)
        if delta >= args.delta_threshold:
            seam_points.append((uv_a, delta))

    deltas = np.array(deltas) if deltas else np.zeros(1)
    overlay = (atlas * 255).astype(np.uint8).copy()
    for point, delta in seam_points:
        x = int(np.clip(point[0] * (size - 1), 0, size - 1))
        y = int(np.clip(point[1] * (size - 1), 0, size - 1))
        cv2.circle(overlay, (x, y), 6, (255, 0, 0), -1)
    cv2.imwrite(args.output, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    report = {
        "mesh": args.mesh,
        "basecolor": args.basecolor,
        "cross_chart_edges": int(len(deltas)),
        "delta_threshold": args.delta_threshold,
        "visible_seam_edges": int(len(seam_points)),
        "visible_seam_percent": round(float(len(seam_points) / max(len(deltas), 1) * 100), 3),
        "delta_mean": round(float(deltas.mean()), 4),
        "delta_p50": round(float(np.percentile(deltas, 50)), 4),
        "delta_p95": round(float(np.percentile(deltas, 95)), 4),
        "delta_max": round(float(deltas.max()), 4),
        "overlay": args.output,
    }
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"SEAMS cross_chart_edges={report['cross_chart_edges']} "
        f"visible={report['visible_seam_edges']} ({report['visible_seam_percent']}%) "
        f"delta p50={report['delta_p50']} p95={report['delta_p95']} max={report['delta_max']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
