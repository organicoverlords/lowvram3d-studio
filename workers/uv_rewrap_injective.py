"""Rebuild TEXCOORD_0 as a globally injective atlas layout.

The atlas this replaces could not carry per-triangle provenance at all. Measured on the
production mesh at 2048: the sum of UV triangle areas was 1.062e8 texels against an atlas
holding 4.19e6, 18,621 triangles collapsed onto four distinct UV triples, and under a
strict-interior test (barycentric > 0.05, so no result can be a shared-seam artifact) 80.8%
of covered texels were claimed by two or more triangles. A single-owner atlas resolves each
texel to one triangle, so every other claimant renders a colour computed for a surface it is
not part of - which is what put the front face on the back of the head. No fusion policy,
protected region, or targeted chart move can repair a layout that is non-injective by area.

So the layout is rebuilt rather than patched. xatlas produces charts that do not overlap by
construction, and the rebuild is gated here on a strict-interior double-claim census rather
than on trusting that guarantee.

World-space positions, vertex normals and triangle topology are carried through unchanged.
Only UV seam vertices are duplicated: every output vertex is a gather from an input vertex, so
positions are copied bit-for-bit, and the output index buffer is checked triangle-for-triangle
against the input before anything is written.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from atlas_raster import injectivity
from mesh_io import read_glb, write_glb


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def geometry_fingerprint(positions: np.ndarray, tris: np.ndarray) -> str:
    """Order-independent digest of the actual surface, not of a buffer layout.

    A rewrap is allowed to renumber vertices, so a byte hash of the position buffer is not the
    right invariant. The set of world-space triangles is.
    """
    corners = np.ascontiguousarray(positions, np.float32)[tris]
    rolled = np.sort(corners.reshape(len(tris), 9).view(np.uint32).reshape(len(tris), 9), axis=1)
    return hashlib.sha256(np.sort(rolled, axis=0).tobytes()).hexdigest()


def rewrap(mesh: Path, output: Path, resolution: int, padding: int,
           report_path: Path) -> dict:
    import xatlas

    started = time.time()
    # Digest the input before the unwrap, not while writing the receipt: the unwrap takes
    # minutes and the receipt must not be the step that discovers the input is unreadable.
    input_sha = sha256(mesh)
    positions, normals, uv, tris = read_glb(mesh)
    positions = np.ascontiguousarray(positions, np.float32)
    normals = np.ascontiguousarray(normals, np.float32)
    tris = np.ascontiguousarray(tris, np.uint32)
    before = geometry_fingerprint(positions, tris.astype(np.int64))

    previous = None
    if uv is not None and len(uv):
        previous = injectivity(np.asarray(uv, np.float64), tris.astype(np.int64), 2048)

    atlas = xatlas.Atlas()
    atlas.add_mesh(positions, tris)
    chart_options = xatlas.ChartOptions()
    pack_options = xatlas.PackOptions()
    pack_options.resolution = int(resolution)
    pack_options.padding = int(padding)
    pack_options.bruteForce = False
    pack_options.blockAlign = True
    atlas.generate(chart_options, pack_options)
    unwrap_seconds = time.time() - started
    if atlas.atlas_count != 1:
        raise RuntimeError(f"UV_REWRAP_MULTIPLE_ATLASES:{atlas.atlas_count}")

    vertex_map, new_tris, new_uv = atlas[0]
    vertex_map = np.asarray(vertex_map, np.int64)
    new_tris = np.ascontiguousarray(new_tris, np.int64)
    new_uv = np.asarray(new_uv, np.float64)
    if float(new_uv.max()) > 1.5:  # xatlas reports UVs in texels when a resolution is packed
        new_uv = new_uv / np.array([float(atlas.width), float(atlas.height)], np.float64)
    new_uv = np.clip(new_uv, 0.0, 1.0)

    # Topology gate: same triangles, same order, same corners, before anything is written.
    if len(new_tris) != len(tris):
        raise RuntimeError(f"UV_REWRAP_TRIANGLE_COUNT_CHANGED:{len(new_tris)}:{len(tris)}")
    if not np.array_equal(vertex_map[new_tris], tris.astype(np.int64)):
        raise RuntimeError("UV_REWRAP_TRIANGLE_CORNERS_CHANGED")

    new_positions = positions[vertex_map]
    new_normals = normals[vertex_map]
    after = geometry_fingerprint(new_positions, new_tris)
    if after != before:
        raise RuntimeError("UV_REWRAP_GEOMETRY_FINGERPRINT_CHANGED")

    gate = injectivity(new_uv, new_tris, int(resolution))
    if not gate["injective"]:
        raise RuntimeError(
            f"UV_REWRAP_NOT_INJECTIVE:{gate['interior_texels_claimed_twice']}")
    if gate["analytic_uv_area_fraction"] > 1.0:
        raise RuntimeError(
            f"UV_REWRAP_AREA_EXCEEDS_ATLAS:{gate['analytic_uv_area_fraction']}")

    output.parent.mkdir(parents=True, exist_ok=True)
    write_glb(output, new_positions, new_normals, new_uv.astype(np.float32), new_tris)

    report = {
        "schema": "uv_rewrap_injective_v1",
        "packer": f"xatlas {getattr(xatlas, '__version__', 'unknown')}",
        "input_mesh": str(mesh),
        "input_mesh_sha256": input_sha,
        "output_mesh": str(output),
        "output_mesh_sha256": sha256(output),
        "atlas_resolution": int(resolution),
        "atlas_count": int(atlas.atlas_count),
        "packed_width": int(atlas.width),
        "packed_height": int(atlas.height),
        "chart_count": int(atlas.chart_count),
        "padding_texels": int(padding),
        "triangles": int(len(new_tris)),
        "vertices_in": int(len(positions)),
        "vertices_out": int(len(new_positions)),
        "seam_vertices_added": int(len(new_positions) - len(positions)),
        "geometry_fingerprint": before,
        "geometry_preserved": True,
        "topology_preserved": True,
        "uv_hash_preserved": False,
        "unwrap_seconds": round(unwrap_seconds, 1),
        "injectivity_after": gate,
        "injectivity_before_at_2048": previous,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    parser.add_argument("--resolution", type=int, default=4096)
    parser.add_argument("--padding", type=int, default=4)
    args = parser.parse_args()
    report = rewrap(Path(args.mesh), Path(args.output), args.resolution, args.padding,
                    Path(args.report))
    print(json.dumps(report, indent=2), flush=True)
    print(f"UV_REWRAP_DONE charts={report['chart_count']} "
          f"injective={report['injectivity_after']['injective']} "
          f"utilisation={report['injectivity_after']['analytic_uv_area_fraction']:.4f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
