"""Gate a UV-mapped GLB with the repository's exact positive-area overlap detector.

This exists because two cheaper metrics were used as a gate and both were wrong: rasterised
pixel-collision counts charge the shared edge between adjacent triangles as overlap, and summed
analytic UV area against rasterised coverage conflates boundary rounding with genuine overlap.
`lowvram3d.uv_overlap` says so explicitly and clips triangles exactly instead. Only that result may
decide pass or fail; the stretch and utilisation figures here are diagnostics reported alongside.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from lowvram3d.uv_overlap import positive_area_uv_overlaps  # noqa: E402
from lowvram3d.uv_quality import (  # noqa: E402
    MAX_CHART_COUNT,
    MAX_STRETCH_P95,
    MAX_TINY_CHART_SURFACE_PERCENT,
    MIN_ATLAS_UTILIZATION,
    area_weighted_percentile,
    conformal_stretch,
)


_COMPONENT = {5120: "i1", 5121: "u1", 5122: "i2", 5123: "u2", 5125: "u4", 5126: "f4"}
_COUNT = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}


def _accessor(gltf: dict, blob: bytes, index: int) -> np.ndarray:
    accessor = gltf["accessors"][index]
    view = gltf["bufferViews"][accessor["bufferView"]]
    dtype = np.dtype(_COMPONENT[accessor["componentType"]]).newbyteorder("<")
    columns = _COUNT[accessor["type"]]
    start = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    count = accessor["count"]
    stride = view.get("byteStride") or dtype.itemsize * columns
    if stride == dtype.itemsize * columns:
        flat = np.frombuffer(blob, dtype=dtype, count=count * columns, offset=start)
        return flat.reshape(count, columns)
    # Interleaved buffer view: walk the stride explicitly.
    rows = [
        np.frombuffer(blob, dtype=dtype, count=columns, offset=start + row * stride)
        for row in range(count)
    ]
    return np.asarray(rows)


def load_uv_mesh(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read POSITION, TEXCOORD_0 and indices straight out of the GLB.

    trimesh exposes ColorVisuals for a mesh with vertex colours and no material, and silently drops
    TEXCOORD_0 in that case, so the UV layer has to come from the glTF accessors directly.
    """
    import struct

    data = path.read_bytes()
    if data[:4] != b"glTF":
        raise RuntimeError(f"{path} is not a binary glTF")
    json_length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + json_length])
    chunk = 20 + json_length
    padding = (4 - json_length % 4) % 4
    chunk += padding
    bin_length = struct.unpack("<I", data[chunk : chunk + 4])[0]
    blob = data[chunk + 8 : chunk + 8 + bin_length]

    positions: list[np.ndarray] = []
    uvs: list[np.ndarray] = []
    for mesh in gltf["meshes"]:
        for primitive in mesh["primitives"]:
            attributes = primitive["attributes"]
            if "TEXCOORD_0" not in attributes:
                continue
            position = _accessor(gltf, blob, attributes["POSITION"]).astype(np.float64)
            uv = _accessor(gltf, blob, attributes["TEXCOORD_0"]).astype(np.float64)
            indices = _accessor(gltf, blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
            positions.append(position[indices])
            uvs.append(uv[indices])
    if not uvs:
        raise RuntimeError(f"{path} carries no TEXCOORD_0")
    return np.concatenate(positions), np.concatenate(uvs)


def atlas_utilisation(uv_triangles: np.ndarray, resolution: int) -> float:
    """Coverage of the atlas, rasterised without any inset so the number is comparable to the
    repository's MIN_ATLAS_UTILIZATION."""
    grid = np.zeros((resolution, resolution), bool)
    for tri in uv_triangles:
        pts = tri * (resolution - 1)
        x0, y0 = np.floor(pts.min(axis=0)).astype(int)
        x1, y1 = np.ceil(pts.max(axis=0)).astype(int)
        x0 = max(x0, 0); y0 = max(y0, 0)
        x1 = min(x1, resolution - 1); y1 = min(y1, resolution - 1)
        if x1 < x0 or y1 < y0:
            continue
        area = (pts[1, 0] - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[2, 0] - pts[0, 0]) * (
            pts[1, 1] - pts[0, 1]
        )
        if abs(area) < 1e-12:
            continue
        ys, xs = np.mgrid[y0 : y1 + 1, x0 : x1 + 1]
        px, py = xs + 0.5, ys + 0.5
        w0 = ((pts[1, 0] - pts[0, 0]) * (py - pts[0, 1]) - (px - pts[0, 0]) * (pts[1, 1] - pts[0, 1])) / area
        w1 = ((px - pts[0, 0]) * (pts[2, 1] - pts[0, 1]) - (pts[2, 0] - pts[0, 0]) * (py - pts[0, 1])) / area
        inside = (w0 >= 0) & (w1 >= 0) & (w0 + w1 <= 1)
        if inside.any():
            grid[ys[inside], xs[inside]] = True
    return float(grid.mean())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--timeout-seconds", type=float, default=60.0)
    args = parser.parse_args()

    path = Path(args.input)
    positions, uv_triangles = load_uv_mesh(path)

    started = time.monotonic()
    overlap = positive_area_uv_overlaps(
        uv_triangles, args.resolution, timeout_seconds=args.timeout_seconds
    )
    overlap_seconds = time.monotonic() - started

    stretch, area3d = conformal_stretch(positions, uv_triangles)
    stretch_p95 = area_weighted_percentile(stretch, area3d, 95.0)
    utilisation = atlas_utilisation(uv_triangles, args.resolution)

    report = overlap.as_dict()
    report.update(
        {
            "input": str(path),
            "atlas_resolution": args.resolution,
            "triangles": int(len(uv_triangles)),
            "atlas_utilization": utilisation,
            "atlas_count": 1,
            "stretch_p95": stretch_p95,
            "overlap_runtime_seconds": overlap_seconds,
            "thresholds": {
                "MIN_ATLAS_UTILIZATION": MIN_ATLAS_UTILIZATION,
                "MAX_CHART_COUNT": MAX_CHART_COUNT,
                "MAX_STRETCH_P95": MAX_STRETCH_P95,
                "MAX_TINY_CHART_SURFACE_PERCENT": MAX_TINY_CHART_SURFACE_PERCENT,
            },
        }
    )
    gate = {
        "exact_overlap_validation_succeeded": overlap.success,
        "no_positive_overlap": overlap.positive_overlap_pair_count == 0,
        "no_degenerate": overlap.degenerate_uv_triangle_count == 0,
        "no_out_of_bounds": overlap.out_of_bounds_triangle_count == 0,
        "utilization_ok": utilisation >= MIN_ATLAS_UTILIZATION,
        "stretch_ok": bool(np.isfinite(stretch_p95) and stretch_p95 <= MAX_STRETCH_P95),
    }
    report["gate"] = gate
    report["gate_passed"] = all(gate.values())

    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(
        f"EXACT_UV candidate_pairs={overlap.candidate_pair_count} tested={overlap.tested_pair_count} "
        f"positive_pairs={overlap.positive_overlap_pair_count} "
        f"texels={overlap.positive_overlap_total_texels_equivalent:.4f} "
        f"degenerate={overlap.degenerate_uv_triangle_count} oob={overlap.out_of_bounds_triangle_count} "
        f"util={utilisation*100:.2f}% stretch_p95={stretch_p95:.3f} "
        f"timed_out={overlap.timed_out} success={overlap.success} PASS={report['gate_passed']}",
        flush=True,
    )
    if overlap.errors:
        print("errors: " + "; ".join(overlap.errors), file=sys.stderr, flush=True)
    return 0 if report["gate_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
