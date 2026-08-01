"""Re-unwrap LOD0 with xatlas and select a candidate by the repository's lexicographic rule.

Blender's Smart UV Project left 123983 positive-area overlapping triangle pairs on this mesh
(~922k texel-equivalents at 4096), which would bake competing surfaces into the same texels.
xatlas charts and packs without needing semantic part names, and handles a single object holding
many disconnected components, which is exactly the shaman's shape.

Candidate metrics and the lexicographic selection come from `lowvram3d.uv_quality`; the pass/fail
overlap number comes from `lowvram3d.uv_overlap`. Nothing here invents a weighted score, and no
positions or normals are modified - xatlas only splits vertices along chart seams.
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import xatlas

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "workers"))

from lowvram3d.uv_overlap import positive_area_uv_overlaps  # noqa: E402
from lowvram3d.uv_quality import (  # noqa: E402
    MAX_CHART_COUNT,
    MAX_STRETCH_P95,
    MAX_TINY_CHART_SURFACE_PERCENT,
    MIN_ATLAS_UTILIZATION,
    PRESETS,
    UvCandidateMetrics,
    area_weighted_percentile,
    conformal_stretch,
    select_candidate,
)
from uv_exact_validate import _accessor, atlas_utilisation  # noqa: E402


def load_indexed(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = path.read_bytes()
    json_length = struct.unpack("<I", data[12:16])[0]
    gltf = json.loads(data[20 : 20 + json_length])
    chunk = 20 + json_length + ((4 - json_length % 4) % 4)
    bin_length = struct.unpack("<I", data[chunk : chunk + 4])[0]
    blob = data[chunk + 8 : chunk + 8 + bin_length]
    primitive = gltf["meshes"][0]["primitives"][0]
    positions = _accessor(gltf, blob, primitive["attributes"]["POSITION"]).astype(np.float64)
    normals = (
        _accessor(gltf, blob, primitive["attributes"]["NORMAL"]).astype(np.float64)
        if "NORMAL" in primitive["attributes"]
        else None
    )
    indices = _accessor(gltf, blob, primitive["indices"]).astype(np.int64).reshape(-1, 3)
    return positions, normals, indices


def weld(positions: np.ndarray, indices: np.ndarray, decimals: int = 6):
    """glTF stores per-corner vertices; xatlas needs the welded surface or every triangle becomes
    its own chart."""
    keys = np.round(positions, decimals)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    return positions[first], inverse[indices], first


def chart_count_from_topology(indices: np.ndarray, vertex_count: int) -> int:
    """xatlas splits vertices along seams, so connected components of the output index buffer are
    exactly the charts."""
    parent = np.arange(vertex_count)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv
    return len({find(i) for i in range(vertex_count)})


def tiny_chart_surface_percent(
    indices: np.ndarray, vertex_count: int, uv: np.ndarray, positions: np.ndarray, resolution: int
) -> float:
    """Share of real surface area sitting in charts too small to carry usable texels."""
    parent = np.arange(vertex_count)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for tri in indices:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c)):
            ru, rv = find(u), find(v)
            if ru != rv:
                parent[ru] = rv

    tri_pos = positions[indices]
    area3d = 0.5 * np.linalg.norm(
        np.cross(tri_pos[:, 1] - tri_pos[:, 0], tri_pos[:, 2] - tri_pos[:, 0]), axis=1
    )
    tri_uv = uv[indices]
    area_uv = 0.5 * np.abs(
        (tri_uv[:, 1, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 2, 1] - tri_uv[:, 0, 1])
        - (tri_uv[:, 2, 0] - tri_uv[:, 0, 0]) * (tri_uv[:, 1, 1] - tri_uv[:, 0, 1])
    )
    roots = np.array([find(int(t[0])) for t in indices])
    tiny_area = 0.0
    for root in np.unique(roots):
        mask = roots == root
        texels = area_uv[mask].sum() * resolution * resolution
        # Below an 8x8 texel footprint a chart cannot hold meaningful baked detail.
        if texels < 64.0:
            tiny_area += float(area3d[mask].sum())
    return 100.0 * tiny_area / max(float(area3d.sum()), 1e-20)


def write_glb(path: Path, positions: np.ndarray, normals, uv: np.ndarray, indices: np.ndarray) -> None:
    pos = positions.astype("<f4")
    uvs = uv.astype("<f4")
    idx = indices.astype("<u4").reshape(-1)
    nrm = normals.astype("<f4") if normals is not None else None

    buffers = [pos.tobytes(), uvs.tobytes(), idx.tobytes()]
    if nrm is not None:
        buffers.insert(1, nrm.tobytes())
    offsets, cursor, blob = [], 0, b""
    for payload in buffers:
        pad = (4 - len(blob) % 4) % 4
        blob += b"\x00" * pad
        offsets.append(len(blob))
        blob += payload

    views, accessors, attributes = [], [], {}
    order = ["POSITION"] + (["NORMAL"] if nrm is not None else []) + ["TEXCOORD_0"]
    arrays = [pos] + ([nrm] if nrm is not None else []) + [uvs]
    for i, (name, array) in enumerate(zip(order, arrays)):
        views.append({"buffer": 0, "byteOffset": offsets[i], "byteLength": array.nbytes, "target": 34962})
        accessor = {
            "bufferView": i,
            "componentType": 5126,
            "count": int(len(array)),
            "type": "VEC3" if array.shape[1] == 3 else "VEC2",
        }
        if name == "POSITION":
            accessor["min"] = array.min(axis=0).astype(float).tolist()
            accessor["max"] = array.max(axis=0).astype(float).tolist()
        attributes[name] = len(accessors)
        accessors.append(accessor)
    views.append({"buffer": 0, "byteOffset": offsets[-1], "byteLength": idx.nbytes, "target": 34963})
    accessors.append(
        {"bufferView": len(views) - 1, "componentType": 5125, "count": int(len(idx)), "type": "SCALAR"}
    )

    gltf = {
        "asset": {"version": "2.0", "generator": "lowvram3d uv_xatlas_route"},
        "scene": 0,
        "scenes": [{"nodes": [0]}],
        "nodes": [{"mesh": 0, "name": "shaman_lod0_uv"}],
        "meshes": [
            {"name": "geometry_0", "primitives": [{"attributes": attributes, "indices": len(accessors) - 1}]}
        ],
        "buffers": [{"byteLength": len(blob)}],
        "bufferViews": views,
        "accessors": accessors,
    }
    raw = json.dumps(gltf, separators=(",", ":")).encode()
    raw += b" " * ((4 - len(raw) % 4) % 4)
    blob += b"\x00" * ((4 - len(blob) % 4) % 4)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        handle.write(b"glTF" + struct.pack("<II", 2, 12 + 8 + len(raw) + 8 + len(blob)))
        handle.write(struct.pack("<II", len(raw), 0x4E4F534A) + raw)
        handle.write(struct.pack("<II", len(blob), 0x004E4942) + blob)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--padding", type=int, default=4)
    parser.add_argument("--overlap-timeout", type=float, default=3000.0)
    args = parser.parse_args()

    source = Path(args.input)
    positions_raw, normals_raw, indices_raw = load_indexed(source)
    positions, indices, first = weld(positions_raw, indices_raw)
    normals = normals_raw[first] if normals_raw is not None else None
    print(f"welded: {len(positions_raw)} -> {len(positions)} vertices, {len(indices)} faces", flush=True)

    candidates: list[UvCandidateMetrics] = []
    results: dict[str, dict] = {}

    for preset in PRESETS:
        started = time.monotonic()
        atlas = xatlas.Atlas()
        atlas.add_mesh(positions, indices.astype(np.uint32))
        chart_options = xatlas.ChartOptions()
        chart_options.max_cost = preset.max_cost
        chart_options.max_iterations = preset.max_iterations
        pack_options = xatlas.PackOptions()
        pack_options.resolution = args.resolution
        pack_options.padding = args.padding
        pack_options.bruteForce = False
        atlas.generate(chart_options=chart_options, pack_options=pack_options)
        vmapping, out_indices, out_uv = atlas[0]
        runtime = time.monotonic() - started

        out_uv = np.asarray(out_uv, np.float64)
        if out_uv.max() > 1.5:  # xatlas may return texel coordinates
            out_uv = out_uv / np.array([atlas.width, atlas.height], np.float64)
        out_indices = np.asarray(out_indices, np.int64).reshape(-1, 3)
        out_positions = positions[vmapping]
        out_normals = normals[vmapping] if normals is not None else None

        uv_triangles = out_uv[out_indices]
        pos_triangles = out_positions[out_indices]

        overlap = positive_area_uv_overlaps(
            uv_triangles, args.resolution, timeout_seconds=args.overlap_timeout
        )
        stretch, area3d = conformal_stretch(pos_triangles, uv_triangles)
        stretch_p95 = area_weighted_percentile(stretch, area3d, 95.0)
        # xatlas 0.0.11 exposes utilization as a scalar; older builds return one entry per atlas.
        raw_utilisation = getattr(atlas, "utilization", None)
        if isinstance(raw_utilisation, (list, tuple, np.ndarray)) and len(raw_utilisation):
            utilisation = float(raw_utilisation[0])
        elif isinstance(raw_utilisation, float):
            utilisation = float(raw_utilisation)
        else:
            utilisation = atlas_utilisation(uv_triangles, args.resolution)
        charts = int(atlas.get_mesh_chart_count(0)) if hasattr(atlas, "get_mesh_chart_count") else chart_count_from_topology(out_indices, len(out_positions))
        tiny = tiny_chart_surface_percent(
            out_indices, len(out_positions), out_uv, out_positions, args.resolution
        )

        errors: list[str] = []
        if atlas.atlas_count != 1:
            errors.append(f"atlas_count={atlas.atlas_count}, requires exactly 1")
        if not overlap.success:
            errors.append("exact overlap validation did not succeed: " + "; ".join(overlap.errors))
        if overlap.positive_overlap_pair_count:
            errors.append(f"{overlap.positive_overlap_pair_count} positive-area overlapping pairs")
        if overlap.degenerate_uv_triangle_count:
            errors.append(f"{overlap.degenerate_uv_triangle_count} degenerate UV triangles")
        if overlap.out_of_bounds_triangle_count:
            errors.append(f"{overlap.out_of_bounds_triangle_count} out-of-bounds triangles")
        if utilisation < MIN_ATLAS_UTILIZATION:
            errors.append(f"utilization {utilisation:.4f} < {MIN_ATLAS_UTILIZATION}")
        if charts > MAX_CHART_COUNT:
            errors.append(f"chart_count {charts} > {MAX_CHART_COUNT}")
        if not np.isfinite(stretch_p95) or stretch_p95 > MAX_STRETCH_P95:
            errors.append(f"stretch_p95 {stretch_p95} > {MAX_STRETCH_P95}")
        if tiny > MAX_TINY_CHART_SURFACE_PERCENT:
            errors.append(f"tiny_chart_surface_percent {tiny:.4f} > {MAX_TINY_CHART_SURFACE_PERCENT}")

        metrics = UvCandidateMetrics(
            preset=preset.name,
            chart_count=charts,
            atlas_utilization=utilisation,
            atlas_count=int(atlas.atlas_count),
            atlas_width=int(atlas.width),
            atlas_height=int(atlas.height),
            overlap_pair_count=overlap.positive_overlap_pair_count,
            overlap_texel_area=overlap.positive_overlap_total_texels_equivalent,
            degenerate_triangle_count=overlap.degenerate_uv_triangle_count,
            out_of_bounds_triangle_count=overlap.out_of_bounds_triangle_count,
            stretch_p95=stretch_p95,
            tiny_chart_surface_percent=tiny,
            runtime_seconds=runtime,
            valid=not errors,
            errors=errors,
            max_cost=preset.max_cost,
        )
        candidates.append(metrics)
        results[preset.name] = {
            "metrics": metrics.as_dict(),
            "exact_overlap": overlap.as_dict(),
            "payload": (out_positions, out_normals, out_uv, out_indices),
        }
        print(
            f"preset {preset.name}: charts={charts} util={utilisation*100:.2f}% "
            f"overlap_pairs={overlap.positive_overlap_pair_count} texels={metrics.overlap_texel_area:.2f} "
            f"degenerate={metrics.degenerate_triangle_count} oob={metrics.out_of_bounds_triangle_count} "
            f"stretch_p95={stretch_p95:.3f} tiny={tiny:.4f}% valid={metrics.valid}",
            flush=True,
        )

    chosen = select_candidate(candidates)
    report = {
        "input": str(source),
        "resolution": args.resolution,
        "padding": args.padding,
        "candidates": [c.as_dict() for c in candidates],
        "exact_overlap_by_preset": {k: v["exact_overlap"] for k, v in results.items()},
        "selection_rule": "lexicographic: chart_count, -utilization, stretch_p95, tiny_chart_surface_percent, max_cost",
        "selected": chosen.preset if chosen else None,
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, indent=2), encoding="utf-8")

    if chosen is None:
        print("XATLAS_NO_VALID_CANDIDATE", file=sys.stderr, flush=True)
        return 1

    out_positions, out_normals, out_uv, out_indices = results[chosen.preset]["payload"]
    write_glb(Path(args.output), out_positions, out_normals, out_uv, out_indices)
    print(
        f"XATLAS_SELECTED preset={chosen.preset} charts={chosen.chart_count} "
        f"util={chosen.atlas_utilization*100:.2f}% stretch_p95={chosen.stretch_p95:.3f} -> {args.output}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

