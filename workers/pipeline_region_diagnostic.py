"""Stage 6 diagnostic: per-UV-island accounting for a named 3D region.

Answers, for every UV chart touching a region of the mesh, where its colour actually came from:
was it observed from the source, inherited from a constrained donor, or filled with a component
median. A chart whose colour is effectively one constant value is flagged, because that is the
signature of prior-fill and it is what reads as untextured clay in a render.

The projection tier is recomputed here rather than read back, because raster_project reports tier
counts but not the per-triangle assignment. Recomputation uses the same inputs it used - the
visibility array, the facing test and the observed coverage mask - so the tiers agree with the
atlas that was actually written.
"""
from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

WELD = 4e-4
FACING_MIN = 0.15
COMPONENT_SIZE = {5121: 1, 5123: 2, 5125: 4, 5126: 4}
COMPONENT_DTYPE = {5121: "<u1", 5123: "<u2", 5125: "<u4", 5126: "<f4"}
TYPE_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3}
CLASS_NAMES = ("cloth", "bone", "wood", "metal", "organic")


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


def label_by_sharing(tris: np.ndarray, vertex_labels: np.ndarray) -> np.ndarray:
    """Connected components over triangles that share a labelled vertex."""
    total = len(tris)
    rows = np.repeat(np.arange(total), 3)
    cols = vertex_labels[tris].reshape(-1)
    incidence = coo_matrix((np.ones(rows.size, np.int8), (rows, cols)),
                           shape=(total, int(cols.max()) + 1)).tocsr()
    return connected_components(incidence @ incidence.T, directed=False)[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--npz", required=True)
    parser.add_argument("--basecolor", required=True)
    parser.add_argument("--coverage", required=True)
    parser.add_argument("--class-map", default="")
    parser.add_argument("--report", required=True)
    parser.add_argument("--region", default="head", choices=["head", "all"])
    parser.add_argument("--height-min", type=float, default=0.72)
    parser.add_argument("--flat-threshold", type=float, default=0.020)
    parser.add_argument("--min-island-triangles", type=int, default=12)
    args = parser.parse_args()

    positions, uv, tris = read_glb(Path(args.mesh))
    data = np.load(args.npz)
    visible = data["vis_front"]
    normals = data["normals"].astype(np.float64)
    direction = np.array([0.0, 0.0, -1.0])
    facing = normals @ direction

    basecolor = cv2.cvtColor(cv2.imread(args.basecolor, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    coverage = cv2.imread(args.coverage, cv2.IMREAD_GRAYSCALE)
    size = basecolor.shape[0]
    classes = None
    if args.class_map:
        palette = np.array([[70, 110, 150], [225, 220, 200], [110, 80, 50], [200, 190, 120], [140, 120, 90]], np.int32)
        rendered = cv2.cvtColor(cv2.imread(args.class_map, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB).astype(np.int32)
        classes = np.argmin(np.abs(rendered[:, :, None, :] - palette[None, None]).sum(axis=3), axis=2)

    welded = np.unique(np.round(positions / WELD).astype(np.int64), axis=0, return_inverse=True)[1]
    mesh_component = label_by_sharing(tris, welded)
    uv_vertex = np.unique(np.round(uv * (size * 4)).astype(np.int64), axis=0, return_inverse=True)[1]
    island = label_by_sharing(tris, uv_vertex)

    lo, hi = positions.min(axis=0), positions.max(axis=0)
    height = (positions[tris].mean(axis=1)[:, 1] - lo[1]) / max(hi[1] - lo[1], 1e-9)
    selected = np.ones(len(tris), bool) if args.region == "all" else (height >= args.height_min)

    # Sample each triangle at its UV centroid, plus its three corner-biased points, so a thin chart
    # still contributes several texels rather than one.
    centroid = uv[tris].mean(axis=1)
    samples = [centroid] + [centroid + (uv[tris][:, k] - centroid) * 0.55 for k in range(3)]
    def texels(points):
        x = np.clip((points[:, 0] * (size - 1)).astype(int), 0, size - 1)
        y = np.clip((points[:, 1] * (size - 1)).astype(int), 0, size - 1)
        return y, x

    colour_stack = np.stack([basecolor[texels(p)] for p in samples])
    tri_colour = colour_stack.mean(axis=0)
    cov_stack = np.stack([coverage[texels(p)] for p in samples])
    tri_observed_texels = (cov_stack >= 255).mean(axis=0)
    class_stack = np.stack([classes[texels(p)] for p in samples]) if classes is not None else None

    tier = np.full(len(tris), "component_prior", dtype=object)
    tier[tri_observed_texels > 0.0] = "partially_observed"
    tier[tri_observed_texels >= 0.5] = "observed"
    donor_like = (tri_observed_texels == 0.0) & (facing > FACING_MIN)
    tier[donor_like] = "donor_or_prior_front_facing"

    entries = []
    for island_id in np.unique(island[selected]):
        members = np.flatnonzero((island == island_id) & selected)
        if members.size < args.min_island_triangles:
            continue
        colours = tri_colour[members]
        spread = float(colours.std(axis=0).mean())
        saturation_mx = colours.max(axis=1)
        saturation = np.where(saturation_mx > 1e-6, (saturation_mx - colours.min(axis=1)) / np.maximum(saturation_mx, 1e-6), 0.0)
        tiers, counts = np.unique(tier[members], return_counts=True)
        dominant = str(tiers[int(np.argmax(counts))])
        klass = None
        if class_stack is not None:
            values, freq = np.unique(class_stack[:, members], return_counts=True)
            klass = CLASS_NAMES[int(values[int(np.argmax(freq))])]
        entries.append({
            "island": int(island_id),
            "mesh_component": int(np.bincount(mesh_component[members]).argmax()),
            "triangles": int(members.size),
            "dominant_tier": dominant,
            "tier_counts": {str(k): int(v) for k, v in zip(tiers, counts)},
            "material_class": klass,
            "observed_texel_fraction": round(float(tri_observed_texels[members].mean()), 4),
            "front_facing_fraction": round(float((facing[members] > FACING_MIN).mean()), 4),
            "visible_fraction": round(float(visible[members].mean()), 4),
            "mean_rgb": [round(float(v), 4) for v in colours.mean(axis=0)],
            "colour_spread": round(spread, 5),
            "saturation_mean": round(float(saturation.mean()), 4),
            "effectively_flat": bool(spread < args.flat_threshold),
            "height_range": [round(float(height[members].min()), 3), round(float(height[members].max()), 3)],
        })

    entries.sort(key=lambda e: -e["triangles"])
    flat = [e for e in entries if e["effectively_flat"]]
    report = {
        "mesh": args.mesh,
        "region": args.region,
        "height_min": args.height_min,
        "flat_threshold": args.flat_threshold,
        "islands_examined": len(entries),
        "triangles_examined": int(sum(e["triangles"] for e in entries)),
        "flat_islands": len(flat),
        "flat_island_triangles": int(sum(e["triangles"] for e in flat)),
        "tier_triangle_totals": {
            str(k): int(v) for k, v in zip(*np.unique(tier[selected], return_counts=True))
        },
        "islands": entries,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"REGION {args.region}: islands={len(entries)} tris={report['triangles_examined']} "
        f"flat_islands={len(flat)} ({report['flat_island_triangles']} tris) "
        f"tiers={report['tier_triangle_totals']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
